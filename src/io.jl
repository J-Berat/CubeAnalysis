function resolve_path(base::AbstractString, path::AbstractString)
    expanded = expanduser(path)
    return normpath(isabspath(expanded) ? expanded : joinpath(base, expanded))
end

function load_config(path::AbstractString)
    absolute = abspath(path)
    isfile(absolute) || error("Parameter file not found: $absolute")
    cfg = TOML.parsefile(absolute)
    cfg["_config_path"] = absolute
    cfg["_config_dir"] = dirname(absolute)
    haskey(cfg, "run") || error("Missing required [run] table in $absolute")
    haskey(cfg, "fields") || error("Missing required [fields.*] tables in $absolute")
    run = cfg["run"]
    haskey(run, "input_dir") || error("Missing run.input_dir")
    haskey(run, "output_dir") || error("Missing run.output_dir")
    run["input_dir"] = resolve_path(cfg["_config_dir"], string(run["input_dir"]))
    run["output_dir"] = resolve_path(cfg["_config_dir"], string(run["output_dir"]))
    return cfg
end

field_label(name::AbstractString, meta::FieldMeta) =
    isempty(meta.unit) ? meta.label : "$(meta.label) [$(meta.unit)]"

function field_metadata(name::AbstractString, spec::AbstractDict, default_colormap::Symbol)
    return FieldMeta(
        string(get(spec, "label", name)),
        string(get(spec, "unit", "")),
        Bool(get(spec, "log", false)),
        Symbol(get(spec, "colormap", String(default_colormap))),
    )
end

function require_cube(name::AbstractString, cube)
    ndims(cube) == 3 || error("Field '$name' must be a 3-D cube; got size $(size(cube))")
    all(size(cube) .>= 2) || error("Field '$name' must have at least two cells per dimension")
    return cube
end

function read_fits_cube(path::AbstractString)
    return FITS(path, "r") do fits
        Float32.(read(fits[1]))
    end
end


function read_hdf5_cube(path::AbstractString, dataset_name::AbstractString,
        permutation::Vector{Int}, scale::Float32, offset::Float32; chunk_mb=64)
    HDF5.h5open(path, "r") do file
        haskey(file, dataset_name) || error("HDF5 dataset '$dataset_name' was not found in $path")
        dataset = file[dataset_name]
        length(size(dataset)) == 3 || error("HDF5 dataset '$dataset_name' must be three-dimensional")
        source_shape = size(dataset)
        output_shape = Tuple(source_shape[d] for d in permutation)
        output = Array{Float32,3}(undef, output_shape)
        bytes_per_plane = prod(source_shape[1:2]) * sizeof(Float32)
        planes = max(1, floor(Int, chunk_mb * 1024^2 / max(bytes_per_plane, 1)))
        output_chunk_dimension = findfirst(==(3), permutation)
        for first_plane in 1:planes:source_shape[3]
            range = first_plane:min(first_plane + planes - 1, source_shape[3])
            chunk = Float32.(dataset[:, :, range])
            (scale != 1 || offset != 0) && (@. chunk = scale * chunk + offset)
            transformed = permutation == [1, 2, 3] ? chunk : permutedims(chunk, Tuple(permutation))
            selectdim(output, output_chunk_dimension, range) .= transformed
        end
        return output
    end
end

function read_cube(path::AbstractString, spec::AbstractDict, name::AbstractString)
    extension = lowercase(splitext(path)[2])
    permutation = Int.(get(spec, "permutation", [1, 2, 3]))
    length(permutation) == 3 && sort(permutation) == [1, 2, 3] ||
        error("fields.$name.permutation must contain dimensions 1, 2, 3 exactly once")
    scale = Float32(get(spec, "scale", 1.0))
    offset = Float32(get(spec, "offset", 0.0))
    data = if extension in (".fits", ".fit", ".fts")
        read_fits_cube(path)
    elseif extension in (".h5", ".hdf5", ".hdf")
        dataset = string(get(spec, "dataset", name))
        return require_cube(name, read_hdf5_cube(path, dataset, permutation, scale, offset;
            chunk_mb=Int(get(spec, "chunk_mb", 64))))
    else
        error("Unsupported cube format '$extension' for '$path'. Use FITS, H5, HDF5, or HDF.")
    end

    if permutation != [1, 2, 3]
        data = permutedims(data, Tuple(permutation))
    end
    (scale != 1 || offset != 0) && (data = @. scale * data + offset)
    return require_cube(name, Array{Float32,3}(data))
end

function load_fields_eager(cfg::AbstractDict)
    input_dir = cfg["run"]["input_dir"]
    isdir(input_dir) || error("Input directory does not exist: $input_dir")
    default_colormap = Symbol(get(get(cfg, "render", Dict()), "colormap", "viridis"))
    fields = Dict{String,AbstractArray{Float32,3}}()
    metadata = Dict{String,FieldMeta}()
    reference_size = nothing

    for name in sort!(collect(keys(cfg["fields"])))
        spec = cfg["fields"][name]
        haskey(spec, "file") || error("Missing fields.$name.file")
        path = resolve_path(input_dir, string(spec["file"]))
        isfile(path) || error("Missing cube file for '$name': $path")
        cube = read_cube(path, spec, name)
        if isnothing(reference_size)
            reference_size = size(cube)
        elseif size(cube) != reference_size
            error("Field '$name' has size $(size(cube)); expected $reference_size")
        end
        fields[name] = cube
        metadata[name] = field_metadata(name, spec, default_colormap)
        enforce_memory_budget(fields, cfg; context="input field '$name'")
    end

    derived = get(cfg, "derived", Dict())
    for spec in get(derived, "vectors", Any[])
        name = string(spec["name"])
        components = string.(spec["components"])
        length(components) == 3 || error("Derived vector '$name' needs three components")
        all(haskey(fields, component) for component in components) ||
            error("Derived vector '$name' references missing components $components")
        factor = Float32(get(spec, "factor", 1.0))
        x, y, z = (fields[component] for component in components)
        fields[name] = VectorMagnitude(x, y, z, factor)
        metadata[name] = field_metadata(name, spec, default_colormap)
    end

    for spec in get(derived, "products", Any[])
        name = string(spec["name"])
        factors = string.(spec["factors"])
        isempty(factors) && error("Derived product '$name' has no factors")
        all(haskey(fields, factor) for factor in factors) ||
            error("Derived product '$name' references missing fields $factors")
        sources = AbstractArray{Float32,3}[fields[factor] for factor in factors]
        fields[name] = ProductField(sources, Float32(get(spec, "factor", 1.0)), reference_size)
        metadata[name] = field_metadata(name, spec, default_colormap)
    end
    return fields, metadata
end

function load_fields(cfg::AbstractDict)
    Bool(get(cfg["run"], "field_by_field", false)) && return load_fields_lazy(cfg)
    return load_fields_eager(cfg)
end

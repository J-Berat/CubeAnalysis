function raw_hdf5_spec(store::LazyFieldStore, name::String)
    haskey(store.cfg["fields"], name) || return nothing
    spec = store.cfg["fields"][name]
    path = resolve_path(store.cfg["run"]["input_dir"], string(spec["file"]))
    lowercase(splitext(path)[2]) in (".h5", ".hdf5", ".hdf") || return nothing
    return (path=path, dataset=string(get(spec, "dataset", name)),
        permutation=Int.(get(spec, "permutation", [1, 2, 3])),
        scale=Float32(get(spec, "scale", 1.0)), offset=Float32(get(spec, "offset", 0.0)),
        chunk_mb=Int(get(spec, "chunk_mb", 64)))
end

function stream_aligned_hdf5(action, store::LazyFieldStore, names::Vector{String}; stride=1)
    specs = [raw_hdf5_spec(store, name) for name in names]
    any(isnothing, specs) && return false
    handles = Dict{String,Any}()
    try
        for spec in specs
            haskey(handles, spec.path) || (handles[spec.path] = HDF5.h5open(spec.path, "r"))
        end
        datasets = [handles[spec.path][spec.dataset] for spec in specs]
        shape = size(first(datasets)); permutation = first(specs).permutation
        all(size(dataset) == shape for dataset in datasets) || return false
        all(spec.permutation == permutation for spec in specs) || return false
        bytes_per_plane = prod(shape[1:2]) * sizeof(Float32) * length(names)
        chunk_mb = minimum(spec.chunk_mb for spec in specs)
        planes = max(1, floor(Int, chunk_mb * 1024^2 / max(bytes_per_plane, 1)))
        for first_plane in 1:planes:shape[3]
            range = first_plane:min(first_plane + planes - 1, shape[3])
            chunks = Array{Float32,3}[]
            for (dataset, spec) in zip(datasets, specs)
                chunk = Float32.(dataset[:, :, range])
                (spec.scale != 1 || spec.offset != 0) &&
                    (@. chunk = spec.scale * chunk + spec.offset)
                push!(chunks, permutation == [1, 2, 3] ? chunk :
                    permutedims(chunk, Tuple(permutation)))
            end
            for index in 1:stride:length(first(chunks))
                action(ntuple(column -> chunks[column][index], length(chunks)))
            end
        end
        return true
    finally
        foreach(close, values(handles))
    end
end

function sampled_field(fields, name::String; stride=1, positive=false)
    if fields isa LazyFieldStore
        values = Float64[]
        streamed = stream_aligned_hdf5(fields, [name]; stride) do row
            value = Float64(row[1])
            isfinite(value) || return
            positive && value <= 0 && return
            push!(values, value)
        end
        streamed && return values
    end
    return finite_sample(fields[name]; stride, positive)
end

function analysis_field_size(fields, name::String)
    if fields isa LazyFieldStore
        spec = raw_hdf5_spec(fields, name)
        if !isnothing(spec)
            source_shape = HDF5.h5open(spec.path, "r") do file
                size(file[spec.dataset])
            end
            return Tuple(source_shape[d] for d in spec.permutation)
        end
    end
    return size(fields[name])
end

function paired_sample_fields(fields, xname, yname, spec, stride, logx, logy)
    mask_name = get(spec, "mask_field", nothing)
    weight_name = get(spec, "weight_field", nothing)
    names = [string(xname), string(yname)]
    !isnothing(mask_name) && push!(names, string(mask_name))
    !isnothing(weight_name) && push!(names, string(weight_name))
    if fields isa LazyFieldStore
        x = Float64[]; y = Float64[]; weights = Float64[]
        mask_min = Float64(get(spec, "mask_min", -Inf)); mask_max = Float64(get(spec, "mask_max", Inf))
        streamed = stream_aligned_hdf5(fields, names; stride) do row
            xv, yv = Float64(row[1]), Float64(row[2])
            isfinite(xv) && isfinite(yv) || return
            logx && xv <= 0 && return; logy && yv <= 0 && return
            column = 3
            if !isnothing(mask_name)
                mask = Float64(row[column]); column += 1
                isfinite(mask) && mask_min <= mask <= mask_max || return
            end
            weight = isnothing(weight_name) ? 1.0 : Float64(row[column])
            isfinite(weight) && weight >= 0 || return
            push!(x, logx ? log10(xv) : xv); push!(y, logy ? log10(yv) : yv)
            push!(weights, weight)
        end
        streamed && !isempty(x) && return x, y, weights
    end
    selection = selection_options(spec, fields)
    return paired_sample(fields[xname], fields[yname], stride, logx, logy; selection...)
end

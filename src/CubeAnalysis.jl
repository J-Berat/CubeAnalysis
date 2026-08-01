module CubeAnalysis

using CairoMakie
using FFTW
using FITSIO
using HDF5
using LaTeXStrings
using LinearAlgebra
using Printf
using Random
using Statistics
using TOML

include("plot_style.jl")
include("geometry.jl")
include("lazy_fields.jl")
include("streaming.jl")
include("spectra.jl")
include("quality.jl")
include("diagnostics.jl")
include("topology.jl")
include("anisotropy.jl")
include("directional_structure.jl")
include("ibanez_alignment.jl")

export run_analysis, load_config, load_fields, read_cube, periodic_gradient, isotropic_spectrum

struct FieldMeta
    label::String
    unit::String
    logscale::Bool
    colormap::Symbol
end

struct AnalysisResult
    output_dir::String
    files::Vector{String}
    fields::Vector{String}
end

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

function requested(cfg, name::AbstractString; default=false)
    return Bool(get(get(cfg, "plots", Dict()), name, default))
end

save_data(cfg) = Bool(get(cfg["run"], "save_data", false))

function selected_fields(settings, fields)
    names = string.(get(settings, "fields", sort!(collect(keys(fields)))))
    missing = filter(name -> !haskey(fields, name), names)
    isempty(missing) || error("Plot settings reference missing fields: $missing")
    return names
end

function finite_sample(A; stride::Int=1, positive::Bool=false)
    stride >= 1 || error("sample_stride must be >= 1")
    values = Float64[]
    sizehint!(values, cld(length(A), stride))
    linear = vec(A)
    @inbounds for index in 1:stride:length(linear)
        value = Float64(linear[index])
        isfinite(value) || continue
        positive && value <= 0 && continue
        push!(values, value)
    end
    return values
end

function transformed_sample(A, logscale::Bool, stride::Int)
    values = finite_sample(A; stride, positive=logscale)
    isempty(values) && error("No finite $(logscale ? "positive " : "")values available")
    return logscale ? log10.(values) : values
end

function transformed_map(A, logscale::Bool)
    map = Float64.(A)
    if logscale
        @inbounds for index in eachindex(map)
            value = map[index]
            map[index] = isfinite(value) && value > 0 ? log10(value) : NaN
        end
    end
    return map
end

function robust_range(values, render)
    finite = filter(isfinite, vec(values))
    isempty(finite) && return (0.0, 1.0)
    qlo = Float64(get(render, "quantile_low", 0.005))
    qhi = Float64(get(render, "quantile_high", 0.995))
    0 <= qlo < qhi <= 1 || error("render quantiles must satisfy 0 <= low < high <= 1")
    lo, hi = quantile(finite, (qlo, qhi))
    if lo == hi
        delta = max(abs(lo) * 1e-5, 1e-6)
        lo -= delta
        hi += delta
    end
    return lo, hi
end

function temporary_sibling(path::AbstractString)
    stem, extension = splitext(path)
    return "$(stem).tmp-$(randstring(10))$(extension)"
end

function atomic_output(writer, path::AbstractString; overwrite=true)
    !overwrite && isfile(path) && error("Output exists and overwrite=false: $path")
    temporary = temporary_sibling(path)
    try
        writer(temporary)
        mv(temporary, path; force=overwrite)
    finally
        isfile(temporary) && rm(temporary)
    end
    return path
end

function write_text_output(writer, path::AbstractString; overwrite=true)
    atomic_output(path; overwrite) do temporary
        open(writer, temporary, "w")
    end
end

function publish_output_directory(staging::AbstractString, destination::AbstractString;
        overwrite=true)
    isfile(destination) && error("Output destination is a file: $destination")
    !overwrite && isdir(destination) && error("Output directory exists and overwrite=false: $destination")
    backup = nothing
    if isdir(destination)
        backup = "$(destination).backup-$(randstring(10))"
        mv(destination, backup)
    end
    try
        mv(staging, destination)
    catch
        !isnothing(backup) && isdir(backup) && !ispath(destination) && mv(backup, destination)
        rethrow()
    end
    !isnothing(backup) && isdir(backup) && rm(backup; recursive=true)
    return destination
end

function save_figure!(files, fig, output_dir, stem, formats; overwrite=true)
    for format in formats
        extension = lowercase(string(format))
        extension in ("png", "pdf", "svg") || error("Unsupported output format '$extension'")
        path = joinpath(output_dir, "$stem.$extension")
        atomic_output(path; overwrite) do temporary
            save(temporary, fig)
        end
        push!(files, path)
    end
end

axis_dimension(axis::AbstractString) =
    lowercase(axis) == "x" ? 1 : lowercase(axis) == "y" ? 2 :
    lowercase(axis) == "z" ? 3 : error("Axis must be x, y, or z; got '$axis'")

function slice_map(cube, axis::AbstractString, position::Real)
    dimension = axis_dimension(axis)
    index = clamp(round(Int, Float64(position) * (size(cube, dimension) - 1)) + 1,
        1, size(cube, dimension))
    return Array(selectdim(cube, dimension, index)), index
end

function projection_map(cube, axis::AbstractString, statistic::AbstractString)
    dimension = axis_dimension(axis)
    lowered = lowercase(statistic)
    projected = if lowered == "mean"
        mean(cube; dims=dimension)
    elseif lowered == "sum"
        sum(cube; dims=dimension)
    elseif lowered == "max"
        maximum(cube; dims=dimension)
    elseif lowered == "std"
        std(cube; dims=dimension, corrected=false)
    else
        error("Projection statistic must be mean, sum, max, or std; got '$statistic'")
    end
    return Array(dropdims(projected; dims=dimension))
end

function length_unit_to_cm(unit::AbstractString)
    normalized = lowercase(strip(unit))
    factors = Dict(
        "cm" => 1.0,
        "m" => 1.0e2,
        "au" => 1.495978707e13,
        "pc" => 3.0856775814913673e18,
        "kpc" => 3.0856775814913673e21,
    )
    haskey(factors, normalized) ||
        error("Cannot convert grid.box_unit '$unit' to centimeters for column density")
    return factors[normalized]
end

function column_density_map(cube, axis::AbstractString, cfg)
    dimension = axis_dimension(axis)
    grid = get(cfg, "grid", Dict())
    spacings = grid_spacings(cube, get(grid, "box_size", 1.0);
        axis_order=string.(get(grid, "axis_order", ["x", "y", "z"])))
    cell_length_cm = spacings[dimension] *
        length_unit_to_cm(string(get(grid, "box_unit", "cm")))
    return projection_map(cube, axis, "sum") .* cell_length_cm
end

function column_density_unit(unit::AbstractString)
    isempty(unit) && return "cm"
    occursin("cm^-3", unit) && return replace(unit, "cm^-3" => "cm^-2")
    return "$unit cm"
end

function write_summary!(files, fields, metadata, output_dir, stride; overwrite=true)
    path = joinpath(output_dir, "cube_summary.csv")
    write_text_output(path; overwrite) do io
        println(io, "field,label,unit,nx,ny,nz,finite_samples,minimum,q01,median,mean,std,q99,maximum")
        for name in sort!(collect(keys(fields)))
            values = sampled_field(fields, name; stride)
            isempty(values) && continue
            meta = metadata[name]
            cube_size = analysis_field_size(fields, name)
            @printf(io, "%s,%s,%s,%d,%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                name, replace(meta.label, ',' => ';'), replace(meta.unit, ',' => ';'),
                cube_size..., length(values), minimum(values), quantile(values, 0.01),
                median(values), mean(values), std(values; corrected=false),
                quantile(values, 0.99), maximum(values))
        end
    end
    push!(files, path)
end

function plot_slices!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "slices", Dict())
    axes = string.(get(settings, "axes", ["x", "y", "z"]))
    positions = Float64.(get(settings, "positions", [0.5]))
    render = get(cfg, "render", Dict())
    for name in selected_fields(settings, fields)
        cube, meta = fields[name], metadata[name]
        fig = Figure(size=(Int(get(render, "width", 1200)),
            max(360, 330 * length(positions))), fontsize=Int(get(render, "fontsize", 16)))
        for (row, position) in enumerate(positions), (column, axis) in enumerate(axes)
            map, index = slice_map(cube, axis, position)
            shown = transformed_map(map, meta.logscale)
            ax = latex_axis(fig[row, column], title="$(axis) = $index",
                xlabel="pixel", ylabel="pixel", aspect=DataAspect())
            lo, hi = robust_range(shown, render)
            hm = heatmap!(ax, shown; colormap=meta.colormap, colorrange=(lo, hi))
            column == length(axes) && latex_colorbar(fig[row, column + 1], hm,
                label=(meta.logscale ? "log10 " : "") * field_label(name, meta))
        end
        latex_layout_label(fig[0, 1:length(axes)], "Slices - $(field_label(name, meta))";
            fontsize=22, tellwidth=false)
        save_figure!(files, fig, output_dir, "slices_$(sanitize(name))", formats; overwrite)
    end
end


function plot_projections!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "projections", Dict())
    axes = string.(get(settings, "axes", ["x", "y", "z"]))
    statistics = string.(get(settings, "statistics", ["mean"]))
    column_density_fields = Set(string.(get(settings, "column_density_fields", ["density"])))
    render = get(cfg, "render", Dict())
    for name in selected_fields(settings, fields)
        cube, meta = fields[name], metadata[name]
        is_column_density = name in column_density_fields
        field_statistics = is_column_density ? ["column density"] : statistics
        fig = Figure(size=(Int(get(render, "width", 1200)),
            max(360, 330 * length(field_statistics))), fontsize=Int(get(render, "fontsize", 16)))
        for (row, statistic) in enumerate(field_statistics), (column, axis) in enumerate(axes)
            map = is_column_density ? column_density_map(cube, axis, cfg) :
                projection_map(cube, axis, statistic)
            shown = transformed_map(map, meta.logscale)
            ax = latex_axis(fig[row, column], title="$(statistic), LOS $(axis)",
                xlabel="pixel", ylabel="pixel", aspect=DataAspect())
            lo, hi = robust_range(shown, render)
            hm = heatmap!(ax, shown; colormap=meta.colormap, colorrange=(lo, hi))
            colorbar_label = is_column_density ?
                "column density [$(column_density_unit(meta.unit))]" : field_label(name, meta)
            column == length(axes) && latex_colorbar(fig[row, column + 1], hm,
                label=(meta.logscale ? "log10 " : "") * colorbar_label)
        end
        figure_label = is_column_density ?
            "Column density [$(column_density_unit(meta.unit))]" : field_label(name, meta)
        latex_layout_label(fig[0, 1:length(axes)], "Projections - $figure_label";
            fontsize=22, tellwidth=false)
        save_figure!(files, fig, output_dir, "projections_$(sanitize(name))", formats; overwrite)
    end
end

function histogram_counts(values, bins::Int)
    bins > 1 || error("Histogram bins must be > 1")
    lo, hi = extrema(values)
    lo == hi && (hi = nextfloat(lo))
    edges = collect(range(lo, hi; length=bins + 1))
    counts = zeros(Float64, bins)
    scale = bins / (hi - lo)
    @inbounds for value in values
        bin = clamp(floor(Int, (value - lo) * scale) + 1, 1, bins)
        counts[bin] += 1
    end
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, counts, edges[2] - edges[1]
end

function histogram_counts(values, edges::AbstractVector)
    bins = length(edges) - 1
    bins > 1 || error("Histogram edges must define at least two bins")
    counts = zeros(Float64, bins)
    lo, hi = first(edges), last(edges)
    scale = bins / (hi - lo)
    @inbounds for value in values
        bin = clamp(floor(Int, (value - lo) * scale) + 1, 1, bins)
        counts[bin] += 1
    end
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    return centers, counts, edges[2] - edges[1]
end

function plot_histograms!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)
    settings = get(cfg, "histograms", Dict())
    bins = Int(get(settings, "bins", 80))
    normalize = Bool(get(settings, "normalize", true))
    for name in selected_fields(settings, fields)
        meta = metadata[name]
        sampled = sampled_field(fields, name; stride, positive=meta.logscale)
        isempty(sampled) && error("No valid samples for histogram '$name'")
        values = meta.logscale ? log10.(sampled) : sampled
        centers, counts, width = histogram_counts(values, bins)
        normalize && (counts ./= sum(counts) * width)
        fig = Figure(size=(760, 520))
        ax = latex_axis(fig[1, 1],
            xlabel=(meta.logscale ? "log10 " : "") * field_label(name, meta),
            ylabel=normalize ? "probability density" : "cells",
            title="Distribution of $(meta.label)")
        barplot!(ax, centers, counts; width=width, color=:steelblue)
        save_figure!(files, fig, output_dir, "histogram_$(sanitize(name))", formats; overwrite)
    end

    for group in get(settings, "groups", Any[])
        group_name = string(group["name"])
        names = string.(group["fields"])
        length(names) >= 2 || error("Histogram group '$group_name' needs at least two fields")
        missing = filter(name -> !haskey(fields, name), names)
        isempty(missing) || error("Histogram group '$group_name' references missing fields: $missing")
        logscales = unique(metadata[name].logscale for name in names)
        length(logscales) == 1 || error("Histogram group '$group_name' mixes linear and logarithmic fields")
        logscale = only(logscales)
        samples = [sampled_field(fields, name; stride, positive=logscale) for name in names]
        any(isempty, samples) && error("Histogram group '$group_name' has a field without valid samples")
        values = logscale ? [log10.(sample) for sample in samples] : samples
        low = minimum(minimum, values); high = maximum(maximum, values)
        low == high && (high = nextfloat(low))
        edges = collect(range(low, high; length=bins + 1))
        units = unique(metadata[name].unit for name in names)
        unit = length(units) == 1 ? only(units) : ""
        xlabel = string(get(group, "xlabel", isempty(unit) ? "field value" : "field value [$unit]"))
        fig = Figure(size=(760, 520))
        ax = latex_axis(fig[1, 1],
            xlabel=(logscale ? "log10 " : "") * xlabel,
            ylabel=normalize ? "probability density" : "cells",
            title=string(get(group, "title", replace(group_name, '_' => ' '))))
        for (index, name) in enumerate(names)
            centers, counts, width = histogram_counts(values[index], edges)
            normalize && (counts ./= sum(counts) * width)
            lines!(ax, centers, counts; linewidth=2.5,
                label=latex_legend_label(metadata[name].label))
        end
        axislegend(ax)
        save_figure!(files, fig, output_dir, "histogram_$(sanitize(group_name))", formats; overwrite)
    end
end

function histogram2d(x, y, bins::Int; weights=nothing)
    xmin, xmax = extrema(x); ymin, ymax = extrema(y)
    xmin == xmax && (xmax = nextfloat(xmin))
    ymin == ymax && (ymax = nextfloat(ymin))
    counts = zeros(Float64, bins, bins)
    xscale, yscale = bins / (xmax - xmin), bins / (ymax - ymin)
    @inbounds for index in eachindex(x, y)
        xb = clamp(floor(Int, (x[index] - xmin) * xscale) + 1, 1, bins)
        yb = clamp(floor(Int, (y[index] - ymin) * yscale) + 1, 1, bins)
        counts[xb, yb] += isnothing(weights) ? 1 : weights[index]
    end
    xedges = collect(range(xmin, xmax; length=bins + 1))
    yedges = collect(range(ymin, ymax; length=bins + 1))
    return xedges, yedges, counts
end

function paired_sample(xcube, ycube, stride, logx, logy; mask=nothing,
        mask_min=-Inf, mask_max=Inf, weights=nothing)
    x, y, w = Float64[], Float64[], Float64[]
    sizehint!(x, cld(length(xcube), stride)); sizehint!(y, cld(length(ycube), stride))
    @inbounds for index in 1:stride:length(xcube)
        xv, yv = Float64(xcube[index]), Float64(ycube[index])
        isfinite(xv) && isfinite(yv) || continue
        if !isnothing(mask)
            mask_value = Float64(mask[index])
            isfinite(mask_value) && mask_min <= mask_value <= mask_max || continue
        end
        weight = isnothing(weights) ? 1.0 : Float64(weights[index])
        isfinite(weight) && weight >= 0 || continue
        logx && xv <= 0 && continue
        logy && yv <= 0 && continue
        push!(x, logx ? log10(xv) : xv)
        push!(y, logy ? log10(yv) : yv)
        push!(w, weight)
    end
    isempty(x) && error("No valid paired samples")
    return x, y, w
end

function selection_options(spec, fields)
    mask_name = get(spec, "mask_field", nothing)
    weight_name = get(spec, "weight_field", nothing)
    !isnothing(mask_name) && !haskey(fields, string(mask_name)) &&
        error("Analysis references missing mask field '$mask_name'")
    !isnothing(weight_name) && !haskey(fields, string(weight_name)) &&
        error("Analysis references missing weight field '$weight_name'")
    return (mask=isnothing(mask_name) ? nothing : fields[string(mask_name)],
        mask_min=Float64(get(spec, "mask_min", -Inf)),
        mask_max=Float64(get(spec, "mask_max", Inf)),
        weights=isnothing(weight_name) ? nothing : fields[string(weight_name)])
end

"""Density in thermal equilibrium for the Koyama--Inutsuka cooling law."""
function thermal_equilibrium_density(temperature::Real)
    temperature > 0 || return NaN
    cooling_over_heating = 1.0e7 * exp(-1.184e5 / (temperature + 1000.0)) +
        1.4e-2 * sqrt(temperature) * exp(-92.0 / temperature)
    return inv(cooling_over_heating)
end

function thermal_equilibrium_curve(nmin::Real, nmax::Real; samples=1200,
        temperature_range=(5.0, 1.0e6))
    samples >= 2 || error("Thermal-equilibrium curve needs at least two samples")
    tmin, tmax = Float64.(temperature_range)
    0 < tmin < tmax || error("thermal_temperature_range must contain two positive increasing values")
    temperatures = 10 .^ range(log10(tmin), log10(tmax); length=samples)
    densities = thermal_equilibrium_density.(temperatures)
    keep = isfinite.(densities) .& (densities .>= nmin) .& (densities .<= nmax)
    return densities[keep], temperatures[keep]
end

axis_values(values, logarithmic) = logarithmic ? log10.(values) : values

function plot_thermal_phase_overlays!(ax, spec, xedges, yedges, logx, logy)
    mode = lowercase(string(get(spec, "thermal_overlay", "none")))
    mode in ("none", "temperature", "pressure") ||
        error("thermal_overlay must be none, temperature, or pressure; got '$mode'")
    mode == "none" && return false

    physical_xmin = logx ? 10.0^first(xedges) : first(xedges)
    physical_xmax = logx ? 10.0^last(xedges) : last(xedges)
    temperature_range = Tuple(Float64.(get(spec, "thermal_temperature_range", [5.0, 1.0e6])))
    densities, temperatures = thermal_equilibrium_curve(physical_xmin, physical_xmax;
        samples=Int(get(spec, "thermal_samples", 1200)), temperature_range)
    if !isempty(densities)
        equilibrium_y = mode == "temperature" ? temperatures : densities .* temperatures
        lines!(ax, axis_values(densities, logx), axis_values(equilibrium_y, logy);
            color=:deepskyblue, linewidth=3.0,
            label=latex_legend_label("thermal equilibrium"))
    end

    xphysical = 10 .^ range(log10(physical_xmin), log10(physical_xmax); length=300)
    for temperature in Float64.(get(spec, "isotherms", [100.0, 1000.0, 8000.0]))
        temperature > 0 || error("Isotherm temperatures must be positive")
        isotherm_y = mode == "temperature" ? fill(temperature, length(xphysical)) :
            xphysical .* temperature
        transformed_y = axis_values(isotherm_y, logy)
        maximum(transformed_y) < first(yedges) && continue
        minimum(transformed_y) > last(yedges) && continue
        lines!(ax, axis_values(xphysical, logx), transformed_y;
            color=:black, linestyle=:dash, linewidth=1.5,
            label=latex_legend_label("T = $(@sprintf("%.4g", temperature)) K"))
    end
    axislegend(ax; position=:rb)
    return true
end

function plot_phase_diagrams!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)
    for spec in get(cfg, "phase_diagrams", Any[])
        name, xname, yname = string(spec["name"]), string(spec["x"]), string(spec["y"])
        haskey(fields, xname) && haskey(fields, yname) ||
            error("Phase diagram '$name' references missing fields")
        logx = Bool(get(spec, "log_x", metadata[xname].logscale))
        logy = Bool(get(spec, "log_y", metadata[yname].logscale))
        bins = Int(get(spec, "bins", 120))
        x, y, weights = paired_sample_fields(fields, xname, yname, spec, stride, logx, logy)
        xe, ye, counts = histogram2d(x, y, bins; weights)
        fig = Figure(size=(820, 620))
        ax = latex_axis(fig[1, 1], title=replace(name, '_' => ' '),
            xlabel=(logx ? "log10 " : "") * field_label(xname, metadata[xname]),
            ylabel=(logy ? "log10 " : "") * field_label(yname, metadata[yname]))
        displayed_counts = map(count -> count > 0 ? log10(count) : NaN, counts)
        hm = heatmap!(ax, xe, ye, displayed_counts; colormap=:magma, nan_color=:white)
        plot_thermal_phase_overlays!(ax, spec, xe, ye, logx, logy)
        xlims!(ax, first(xe), last(xe)); ylims!(ax, first(ye), last(ye))
        latex_colorbar(fig[1, 2], hm, label="log10 N")
        save_figure!(files, fig, output_dir, "phase_$(sanitize(name))", formats; overwrite)
    end
end

function conditional_relation(x, y; bins=30, minimum=100, logx=true, logy=false,
        weights=nothing)
    validx = logx ? filter(>(0), x) : x
    isempty(validx) && error("No valid x values for conditional relation")
    low, high = quantile(validx, (0.001, 0.999))
    low == high && (high = nextfloat(low))
    edges = logx ? 10 .^ range(log10(low), log10(high); length=bins + 1) :
        collect(range(low, high; length=bins + 1))
    count = zeros(Int, bins); weight_sum = zeros(Float64, bins)
    sum1 = zeros(Float64, bins); sum2 = zeros(Float64, bins)
    @inbounds for index in eachindex(x, y)
        xv, yv = x[index], y[index]
        isfinite(xv) && isfinite(yv) || continue
        logy && yv <= 0 && continue
        weight = isnothing(weights) ? 1.0 : weights[index]
        isfinite(weight) && weight >= 0 || continue
        (xv < edges[1] || xv > edges[end]) && continue
        bin = clamp(searchsortedlast(edges, xv), 1, bins)
        count[bin] += 1; weight_sum[bin] += weight
        sum1[bin] += weight * yv; sum2[bin] += weight * yv * yv
    end
    centers = logx ? sqrt.(edges[1:end-1] .* edges[2:end]) :
        0.5 .* (edges[1:end-1] .+ edges[2:end])
    keep = (count .>= minimum) .& (weight_sum .> 0)
    means = sum1[keep] ./ weight_sum[keep]
    deviations = sqrt.(max.(sum2[keep] ./ weight_sum[keep] .- means .^ 2, 0.0))
    return centers[keep], means, deviations, count[keep]
end

function plot_relations!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    for spec in get(cfg, "relations", Any[])
        name, xname, yname = string(spec["name"]), string(spec["x"]), string(spec["y"])
        haskey(fields, xname) && haskey(fields, yname) || error("Relation '$name' references missing fields")
        logx, logy = Bool(get(spec, "log_x", false)), Bool(get(spec, "log_y", false))
        x, y, weights = paired_sample_fields(fields, xname, yname, spec, 1, false, false)
        centers, means, deviations, counts = conditional_relation(x, y;
            bins=Int(get(spec, "bins", 30)),
            minimum=Int(get(spec, "minimum_per_bin", 100)), logx, logy, weights)
        isempty(centers) && (@warn "No populated bins for relation" name; continue)
        if save_data(cfg)
            path = joinpath(output_dir, "relation_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "x_center,samples,y_mean,y_std")
                for index in eachindex(centers)
                    @printf(io, "%.8e,%d,%.8e,%.8e\n", centers[index], counts[index], means[index], deviations[index])
                end
            end
            push!(files, path)
        end
        lower = max.(means .- deviations, logy ? eps(Float64) : -Inf)
        fig = Figure(size=(760, 520))
        ax = latex_axis(fig[1, 1], title=replace(name, '_' => ' '),
            xlabel=field_label(xname, metadata[xname]), ylabel=field_label(yname, metadata[yname]),
            xscale=logx ? log10 : identity, yscale=logy ? log10 : identity)
        band!(ax, centers, lower, means .+ deviations; color=(:steelblue, 0.22))
        scatterlines!(ax, centers, means; color=:steelblue, linewidth=2.5)
        save_figure!(files, fig, output_dir, "relation_$(sanitize(name))", formats; overwrite)
    end
end

function alignment_analysis(scalar, vx, vy, vz, condition; condition_bins=12,
        angle_bins=30, minimum=100, box_size=1.0, periodic=true,
        axis_order=("x", "y", "z"), weighting="gradient", angle_coordinate="cosine",
        bootstrap_replicates=0, bootstrap_seed=1234, bootstrap_max_samples=20_000,
        return_uncertainty=false)
    spacings = grid_spacings(scalar, box_size; axis_order)
    weighting in ("uniform", "gradient") ||
        error("HRO weighting must be 'uniform' or 'gradient'")
    angle_coordinate in ("cosine", "degrees") ||
        error("HRO angle_coordinate must be 'cosine' or 'degrees'")
    condition_values = finite_sample(condition; stride=max(1, length(condition) ÷ 1_000_000), positive=true)
    isempty(condition_values) && error("Alignment condition has no positive finite values")
    lo, hi = quantile(condition_values, (0.001, 0.999))
    edges = 10 .^ range(log10(lo), log10(hi); length=condition_bins + 1)
    counts = zeros(Float64, condition_bins, angle_bins)
    samples = zeros(Int, condition_bins)
    parallel_weight, perpendicular_weight = zeros(condition_bins), zeros(condition_bins)
    rng = MersenneTwister(bootstrap_seed)
    reservoirs = [Tuple{Float32,Float32}[] for _ in 1:condition_bins]
    @inbounds for cartesian in CartesianIndices(scalar)
        i, j, k = Tuple(cartesian)
        c = Float64(condition[cartesian])
        isfinite(c) && edges[1] <= c <= edges[end] || continue
        raw_gradient = gradient_at(scalar, i, j, k, spacings, periodic)
        g1, g2, g3 = Float64.(raw_gradient)
        v1, v2, v3 = Float64(vx[cartesian]), Float64(vy[cartesian]), Float64(vz[cartesian])
        gnorm = sqrt(g1^2 + g2^2 + g3^2); vnorm = sqrt(v1^2 + v2^2 + v3^2)
        gnorm > 0 && vnorm > 0 || continue
        cosine = clamp(abs(g1 * v1 + g2 * v2 + g3 * v3) / (gnorm * vnorm), 0.0, 1.0)
        cbin = clamp(searchsortedlast(edges, c), 1, condition_bins)
        coordinate = angle_coordinate == "cosine" ? cosine : acosd(cosine)
        scaled = angle_coordinate == "cosine" ? coordinate : coordinate / 90
        abin = clamp(floor(Int, scaled * angle_bins) + 1, 1, angle_bins)
        weight = weighting == "gradient" ? gnorm : 1.0
        counts[cbin, abin] += weight; samples[cbin] += 1
        cosine >= cosd(22.5) && (parallel_weight[cbin] += weight)
        cosine <= cosd(67.5) && (perpendicular_weight[cbin] += weight)
        if bootstrap_replicates > 0
            reservoir = reservoirs[cbin]
            observation = (Float32(cosine), Float32(weight))
            if length(reservoir) < bootstrap_max_samples
                push!(reservoir, observation)
            else
                replacement = rand(rng, 1:samples[cbin])
                replacement <= bootstrap_max_samples && (reservoir[replacement] = observation)
            end
        end
    end
    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    angle_centers = angle_coordinate == "cosine" ?
        collect(range(0.5 / angle_bins, 1 - 0.5 / angle_bins; length=angle_bins)) :
        collect(range(45 / angle_bins, 90 - 45 / angle_bins; length=angle_bins))
    zeta = fill(NaN, condition_bins)
    zeta_std = fill(NaN, condition_bins)
    for bin in 1:condition_bins
        total = sum(@view counts[bin, :])
        total > 0 && (@views counts[bin, :] ./= total)
        denominator = parallel_weight[bin] + perpendicular_weight[bin]
        samples[bin] >= minimum && denominator > 0 &&
            (zeta[bin] = (parallel_weight[bin] - perpendicular_weight[bin]) / denominator)
        if bootstrap_replicates > 1 && samples[bin] >= minimum && !isempty(reservoirs[bin])
            reservoir = reservoirs[bin]
            estimates = Float64[]
            sizehint!(estimates, bootstrap_replicates)
            for _ in 1:bootstrap_replicates
                parallel = 0.0; perpendicular = 0.0
                for _ in eachindex(reservoir)
                    cosine, weight = reservoir[rand(rng, eachindex(reservoir))]
                    cosine >= cosd(22.5) && (parallel += weight)
                    cosine <= cosd(67.5) && (perpendicular += weight)
                end
                parallel + perpendicular > 0 &&
                    push!(estimates, (parallel - perpendicular) / (parallel + perpendicular))
            end
            length(estimates) > 1 && (zeta_std[bin] = std(estimates))
        end
    end
    base = (centers, angle_centers, counts, zeta, samples)
    return return_uncertainty ? (base..., zeta_std) : base
end

function plot_alignments!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    periodic = Bool(get(grid, "periodic", true))
    for spec in get(cfg, "alignments", Any[])
        name, scalar_name = string(spec["name"]), string(spec["scalar"])
        vector_names = string.(spec["vector"])
        condition_name = string(get(spec, "condition", scalar_name))
        all(haskey(fields, field) for field in [scalar_name; vector_names; condition_name]) ||
            error("Alignment '$name' references missing fields")
        length(vector_names) == 3 || error("Alignment '$name' needs three vector components")
        weighting = string(get(spec, "weighting", "gradient"))
        angle_coordinate = string(get(spec, "angle_coordinate", "cosine"))
        scalar = fields[scalar_name]
        vector_components = [fields[field] for field in vector_names]
        condition = condition_name == scalar_name ? scalar : fields[condition_name]
        distinct_cubes = condition === scalar ? 4 : 5
        enforce_working_set(cfg, distinct_cubes * sizeof(Float32) * length(scalar);
            context="HRO '$name'")
        centers, angles, histogram, zeta, samples, zeta_std = alignment_analysis(
            scalar, vector_components..., condition;
            condition_bins=Int(get(spec, "condition_bins", 12)),
            angle_bins=Int(get(spec, "angle_bins", 30)),
            minimum=Int(get(spec, "minimum_per_bin", 100)), box_size, periodic,
            axis_order, weighting, angle_coordinate,
            bootstrap_replicates=Int(get(spec, "bootstrap_replicates", 200)),
            bootstrap_seed=Int(get(spec, "bootstrap_seed", 1234)),
            bootstrap_max_samples=Int(get(spec, "bootstrap_max_samples", 20_000)),
            return_uncertainty=true)
        if save_data(cfg)
            path = joinpath(output_dir, "alignment_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "condition_center,samples,zeta,zeta_bootstrap_std")
                for index in eachindex(centers)
                    @printf(io, "%.8e,%d,%.8e,%.8e\n", centers[index], samples[index],
                        zeta[index], zeta_std[index])
                end
            end
            push!(files, path)
        end
        fig = Figure(size=(1050, 520))
        angle_label = angle_coordinate == "cosine" ? "|cos(angle)|" : "angle [deg]"
        ax1 = latex_axis(fig[1, 1], title="HRO ($(weighting) weighting)", xlabel=angle_label,
            ylabel=field_label(condition_name, metadata[condition_name]), yscale=log10)
        hm = heatmap!(ax1, angles, centers, permutedims(histogram); colormap=:viridis)
        latex_colorbar(fig[1, 2], hm, label="normalized gradient weight")
        ax2 = latex_axis(fig[1, 3], title="Alignment parameter", xlabel=field_label(condition_name, metadata[condition_name]),
            ylabel="zeta", xscale=log10)
        hlines!(ax2, [0.0]; color=:gray50, linestyle=:dash)
        valid = findall(isfinite, zeta)
        if !isempty(valid)
            scatterlines!(ax2, centers[valid], zeta[valid]; color=:purple, linewidth=2.5)
            errorbars!(ax2, centers[valid], zeta[valid], zeta_std[valid]; color=:purple)
        end
        ylims!(ax2, -1.05, 1.05)
        latex_layout_label(fig[0, :], replace(name, '_' => ' '); fontsize=22, tellwidth=false)
        save_figure!(files, fig, output_dir, "alignment_$(sanitize(name))", formats; overwrite)
        plot_ibanez_xi!(files, fields, metadata, cfg, spec, output_dir, formats, overwrite)
    end
end

sanitize(name::AbstractString) = lowercase(replace(name, r"[^A-Za-z0-9]+" => "_"))

function write_manifest!(files, cfg, output_dir, fields; overwrite=true, timings=Dict{String,Float64}())
    path = joinpath(output_dir, "analysis_manifest.toml")
    manifest = Dict(
        "source_config" => cfg["_config_path"],
        "input_dir" => cfg["run"]["input_dir"],
        "output_dir" => cfg["run"]["output_dir"],
        "cube_size" => collect(analysis_field_size(fields, first(keys(fields)))),
        "fields" => sort!(collect(keys(fields))),
        "generated_files" => sort!(basename.(files)),
        "julia_version" => string(VERSION),
        "cubeanalysis_version" => "0.2.0",
        "package_versions" => Dict(
            "CairoMakie" => string(pkgversion(CairoMakie)),
            "FFTW" => string(pkgversion(FFTW)),
            "FITSIO" => string(pkgversion(FITSIO)),
            "HDF5" => string(pkgversion(HDF5)),
        ),
        "timings_seconds" => timings,
        "config" => Dict(k => v for (k, v) in cfg if !startswith(string(k), "_")),
    )
    write_text_output(path; overwrite) do io
        TOML.print(io, manifest; sorted=true)
    end
    push!(files, path)
end

function run_analysis_config(cfg::AbstractDict)
    final_output_dir = cfg["run"]["output_dir"]
    atomic_directory = Bool(get(cfg["run"], "atomic_directory", false))
    overwrite = Bool(get(cfg["run"], "overwrite", true))
    atomic_directory && !overwrite && ispath(final_output_dir) &&
        error("Output exists and overwrite=false: $final_output_dir")
    if atomic_directory
        mkpath(dirname(final_output_dir))
        output_dir = mktempdir(dirname(final_output_dir);
            prefix="$(basename(final_output_dir)).tmp-")
    else
        output_dir = final_output_dir
        mkpath(output_dir)
    end
    formats = string.(get(cfg["run"], "formats", ["png"]))
    stride = Int(get(cfg["run"], "sample_stride", 1))
    fields, metadata = load_fields(cfg)
    files = String[]
    timings = Dict{String,Float64}()

    @info "Prepared cube fields" input_dir=cfg["run"]["input_dir"] size=analysis_field_size(fields, first(keys(fields))) fields=sort!(collect(keys(fields))) field_by_field=(fields isa LazyFieldStore)
    stages = Tuple{String,Function}[]
    save_data(cfg) && requested(cfg, "quality"; default=true) && push!(stages,
        ("quality", () -> write_quality_report!(files, fields, metadata, output_dir; overwrite)))
    save_data(cfg) && requested(cfg, "summary"; default=true) && push!(stages,
        ("summary", () -> write_summary!(files, fields, metadata, output_dir, stride; overwrite)))
    requested(cfg, "slices") && push!(stages,
        ("slices", () -> plot_slices!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "projections") && push!(stages,
        ("projections", () -> plot_projections!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "histograms") && push!(stages,
        ("histograms", () -> plot_histograms!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)))
    requested(cfg, "phase_diagrams") && push!(stages,
        ("phase_diagrams", () -> plot_phase_diagrams!(files, fields, metadata, cfg, output_dir, formats, overwrite, stride)))
    requested(cfg, "relations") && push!(stages,
        ("relations", () -> plot_relations!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "power_spectra") && push!(stages,
        ("power_spectra", () -> plot_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    requested(cfg, "vector_spectra") && push!(stages,
        ("vector_spectra", () -> plot_vector_spectra!(files, fields, metadata, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "cross_spectra") && push!(stages,
        ("cross_spectra", () -> plot_cross_spectra!(files, fields, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "alignments") && push!(stages,
        ("alignments", () -> plot_alignments!(files, fields, metadata, cfg, output_dir, formats, overwrite)))
    save_data(cfg) && requested(cfg, "advanced_diagnostics") && push!(stages,
        ("advanced_diagnostics", () -> write_advanced_diagnostics!(files, fields, metadata,
            cfg, output_dir, overwrite)))
    requested(cfg, "topology") && push!(stages,
        ("topology", () -> plot_topology!(files, fields, metadata, cfg,
            output_dir, formats, overwrite)))
    requested(cfg, "anisotropic_spectra") && push!(stages,
        ("anisotropic_spectra", () -> plot_anisotropic_spectra!(files, fields, metadata,
            cfg, output_dir, formats, overwrite)))
    requested(cfg, "directional_structure_functions") && push!(stages,
        ("directional_structure_functions", () -> plot_directional_structure!(files, fields,
            metadata, cfg, output_dir, formats, overwrite)))
    for (index, (name, action)) in enumerate(stages)
        timed_stage!(timings, index, length(stages), name, action)
        fields isa LazyFieldStore && GC.gc(false)
    end
    save_data(cfg) && write_manifest!(files, cfg, output_dir, fields; overwrite, timings)
    if atomic_directory
        relative_files = relpath.(files, Ref(output_dir))
        publish_output_directory(output_dir, final_output_dir; overwrite)
        files = joinpath.(final_output_dir, relative_files)
        output_dir = final_output_dir
    end
    return AnalysisResult(output_dir, files, sort!(collect(keys(fields))))
end

function run_snapshot_series(cfg::AbstractDict)
    snapshots = cfg["snapshots"]
    parent_output = cfg["run"]["output_dir"]
    mkpath(parent_output)
    overwrite = Bool(get(cfg["run"], "overwrite", true))
    results = AnalysisResult[]
    series_files = String[]
    rows = String[]
    for (index, snapshot) in enumerate(snapshots)
        name = string(get(snapshot, "name", @sprintf("snapshot_%04d", index)))
        @info "Running snapshot" index total=length(snapshots) name
        child = deepcopy(cfg)
        delete!(child, "snapshots")
        child["run"]["input_dir"] = resolve_path(cfg["_config_dir"], string(snapshot["input_dir"]))
        subdirectory = string(get(snapshot, "output_subdir", sanitize(name)))
        child["run"]["output_dir"] = joinpath(parent_output, subdirectory)
        result = run_analysis_config(child)
        push!(results, result); append!(series_files, result.files)
        summary = joinpath(result.output_dir, "cube_summary.csv")
        if isfile(summary)
            summary_lines = readlines(summary)
            time = get(snapshot, "time", index)
            for line in summary_lines[2:end]
                push!(rows, "$(replace(name, ',' => ';')),$time,$line")
            end
        end
    end
    if save_data(cfg)
        evolution = joinpath(parent_output, "snapshot_evolution.csv")
        write_text_output(evolution; overwrite) do io
            println(io, "snapshot,time,field,label,unit,nx,ny,nz,finite_samples,minimum,q01,median,mean,std,q99,maximum")
            foreach(line -> println(io, line), rows)
        end
        push!(series_files, evolution)
    end
    fields = sort!(unique!(reduce(vcat, [result.fields for result in results]; init=String[])))
    return AnalysisResult(parent_output, series_files, fields)
end

function run_analysis(config_path::AbstractString)
    cfg = load_config(config_path)
    snapshots = get(cfg, "snapshots", Any[])
    return isempty(snapshots) ? run_analysis_config(cfg) : run_snapshot_series(cfg)
end

end

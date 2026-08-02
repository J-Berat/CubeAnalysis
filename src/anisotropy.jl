function normalized_direction(direction)
    values = Float64.(direction)
    length(values) == 3 || error("Guide-field direction needs three components")
    norm = sqrt(sum(abs2, values))
    isfinite(norm) && norm > 0 || error("Guide-field direction must be finite and non-zero")
    return Tuple(values ./ norm)
end

function mean_guide_direction(fields, names, stride)
    length(names) == 3 || error("guide_fields needs three field names")
    means = Float64[]
    for name in string.(names)
        haskey(fields, name) || error("Missing guide field '$name'")
        values = sampled_field(fields, name; stride)
        isempty(values) && error("Guide field '$name' has no finite samples")
        push!(means, mean(values))
    end
    return normalized_direction(means)
end

function anisotropic_spectrum(field, direction; parallel_bins=32, perpendicular_bins=32,
        radial_bins=32, box_size=1.0, axis_order=("x", "y", "z"), remove_mean=true)
    guide = normalized_direction(direction)
    workspace = SpectrumWorkspace(size(field))
    prepare_spectrum_input!(workspace, field; remove_mean, window="none")
    shape = size(field); lengths = box_lengths(box_size, axis_order)
    mode_axes = (FFTW.rfftfreq(shape[1]) .* shape[1], FFTW.fftfreq(shape[2]) .* shape[2],
        FFTW.fftfreq(shape[3]) .* shape[3])
    wave = ntuple(d -> 2π .* mode_axes[d] ./ lengths[d], 3)
    kmin = minimum(2π ./ collect(lengths)); kmax = minimum(π .* collect(shape) ./ collect(lengths))
    parallel_edges = collect(range(0, kmax; length=parallel_bins + 1))
    perpendicular_edges = collect(range(0, kmax; length=perpendicular_bins + 1))
    radial_edges = 10 .^ range(log10(kmin), log10(kmax); length=radial_bins + 1)
    power2d = zeros(Float64, parallel_bins, perpendicular_bins)
    modes2d = zeros(Int, parallel_bins, perpendicular_bins)
    parallel_power = zeros(Float64, radial_bins); perpendicular_power = zeros(Float64, radial_bins)
    parallel_modes = zeros(Int, radial_bins); perpendicular_modes = zeros(Int, radial_bins)
    @inbounds for k in axes(workspace.output, 3), j in axes(workspace.output, 2), i in axes(workspace.output, 1)
        kvector = (wave[1][i], wave[2][j], wave[3][k])
        radius = sqrt(sum(abs2, kvector)); kmin <= radius <= kmax || continue
        kparallel = abs(sum(kvector[d] * guide[d] for d in 1:3))
        kperpendicular = sqrt(max(radius^2 - kparallel^2, 0.0))
        pbin = clamp(searchsortedlast(parallel_edges, kparallel), 1, parallel_bins)
        tbin = clamp(searchsortedlast(perpendicular_edges, kperpendicular), 1, perpendicular_bins)
        rbin = clamp(searchsortedlast(radial_edges, radius), 1, radial_bins)
        weight = (i == 1 || (iseven(shape[1]) && i == size(workspace.output, 1))) ? 1 : 2
        power = weight * abs2(workspace.output[i, j, k])
        power2d[pbin, tbin] += power; modes2d[pbin, tbin] += weight
        cosine = kparallel / radius
        if cosine >= 0.8
            parallel_power[rbin] += power; parallel_modes[rbin] += weight
        elseif cosine <= 0.2
            perpendicular_power[rbin] += power; perpendicular_modes[rbin] += weight
        end
    end
    valid = modes2d .> 0
    power2d[valid] ./= modes2d[valid]
    power2d[.!valid] .= NaN
    radial_centers = sqrt.(radial_edges[1:end-1] .* radial_edges[2:end])
    parallel_average = parallel_power ./ max.(parallel_modes, 1)
    perpendicular_average = perpendicular_power ./ max.(perpendicular_modes, 1)
    return (parallel_centers=0.5 .* (parallel_edges[1:end-1] .+ parallel_edges[2:end]),
        perpendicular_centers=0.5 .* (perpendicular_edges[1:end-1] .+ perpendicular_edges[2:end]),
        power=power2d, modes=modes2d, k=radial_centers,
        parallel=parallel_average, perpendicular=perpendicular_average,
        parallel_modes, perpendicular_modes, direction=guide)
end

function plot_anisotropic_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict()); box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    stride = Int(get(cfg["run"], "sample_stride", 1))
    for spec in get(cfg, "anisotropic_spectra", Any[])
        name = string(spec["name"]); field_name = string(spec["field"])
        haskey(fields, field_name) || error("Anisotropic spectrum '$name' references missing field")
        direction = haskey(spec, "direction") ? normalized_direction(spec["direction"]) :
            mean_guide_direction(fields, spec["guide_fields"], stride)
        field = fields[field_name]
        enforce_working_set(cfg, 12 * length(field); context="anisotropic FFT '$name'")
        result = anisotropic_spectrum(field, direction;
            parallel_bins=Int(get(spec, "parallel_bins", 32)),
            perpendicular_bins=Int(get(spec, "perpendicular_bins", 32)),
            radial_bins=Int(get(spec, "radial_bins", 32)), box_size, axis_order,
            remove_mean=Bool(get(spec, "remove_mean", true)))
        if save_data(cfg)
            map_path = joinpath(output_dir, "anisotropic_spectrum_$(sanitize(name)).csv")
            write_text_output(map_path; overwrite) do io
                println(io, "k_parallel,k_perpendicular,power_mean,modes")
                for j in eachindex(result.perpendicular_centers), i in eachindex(result.parallel_centers)
                    @printf(io, "%.8e,%.8e,%.8e,%d\n", result.parallel_centers[i],
                        result.perpendicular_centers[j], result.power[i, j], result.modes[i, j])
                end
            end
            push!(files, map_path)
            wedge_path = joinpath(output_dir, "anisotropic_wedges_$(sanitize(name)).csv")
            write_text_output(wedge_path; overwrite) do io
                println(io, "k,parallel_power,perpendicular_power,parallel_modes,perpendicular_modes,ratio")
                for i in eachindex(result.k)
                    ratio = result.perpendicular[i] > 0 ? result.parallel[i] / result.perpendicular[i] : NaN
                    @printf(io, "%.8e,%.8e,%.8e,%d,%d,%.8e\n", result.k[i], result.parallel[i],
                        result.perpendicular[i], result.parallel_modes[i], result.perpendicular_modes[i], ratio)
                end
            end
            push!(files, wedge_path)
        end
        fig = publication_figure(size=(900, 680)); ax = latex_axis(fig[1, 1],
            xlabel=latexstring("k_{\\parallel}"), ylabel=latexstring("k_{\\perp}"),
            title="$(replace(name, '_' => ' ')); b̂=$(round.(result.direction; digits=3))")
        shown = log10.(result.power)
        finite = filter(isfinite, shown); range = isempty(finite) ? (0.0, 1.0) : extrema(finite)
        hm = heatmap!(ax, result.parallel_centers, result.perpendicular_centers, shown;
            colormap=:magma, colorrange=range)
        latex_colorbar(fig[1, 2], hm, label="log10 mean power", logcoordinates=true)
        save_figure!(files, fig, output_dir, "anisotropic_spectrum_$(sanitize(name))", formats; overwrite)
    end
end

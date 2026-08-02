struct HROXiCurve
    centers::Vector{Float64}
    xi::Vector{Float64}
    uncertainty::Vector{Float64}
    samples::Vector{Int}
end

struct HROXiGlobal
    xi::Float64
    uncertainty::Float64
    samples::Int
end

function density_weighted_velocity_mean(density, component)
    weight_sum = 0.0
    weighted_sum = 0.0
    @inbounds for index in eachindex(density, component)
        weight = Float64(density[index])
        value = Float64(component[index])
        isfinite(weight) && weight > 0 && isfinite(value) || continue
        weight_sum += weight
        weighted_sum += weight * value
    end
    weight_sum > 0 || error("Velocity dispersion has no positive finite density weights")
    return weighted_sum / weight_sum
end

function velocity_dispersion_1d(density, vx, vy, vz)
    means = (density_weighted_velocity_mean(density, vx),
        density_weighted_velocity_mean(density, vy),
        density_weighted_velocity_mean(density, vz))
    weight_sum = 0.0
    variance_sum = 0.0
    @inbounds for index in eachindex(density, vx, vy, vz)
        weight = Float64(density[index])
        values = (Float64(vx[index]), Float64(vy[index]), Float64(vz[index]))
        isfinite(weight) && weight > 0 && all(isfinite, values) || continue
        weight_sum += weight
        variance_sum += weight * sum((values[axis] - means[axis])^2 for axis in 1:3)
    end
    weight_sum > 0 || error("Velocity dispersion has no valid samples")
    return sqrt(variance_sum / (3weight_sum)), means
end

@inline function xi_block_index(index, shape, blocks_per_axis)
    block_size = ntuple(axis -> cld(shape[axis], blocks_per_axis), 3)
    block = ntuple(axis -> clamp(fld(index[axis] - 1, block_size[axis]) + 1,
        1, blocks_per_axis), 3)
    return block[1] + blocks_per_axis * (block[2] - 1) +
        blocks_per_axis^2 * (block[3] - 1)
end

@inline function xi_category(cosine, parallel_cosine, perpendicular_cosine)
    cosine >= parallel_cosine && return 1
    cosine <= perpendicular_cosine && return 2
    return 0
end

@inline function absolute_cosine(first, second)
    first_norm = sqrt(sum(abs2, first)); second_norm = sqrt(sum(abs2, second))
    denominator = first_norm * second_norm
    denominator > 0 || return NaN
    return clamp(abs(sum(first[axis] * second[axis] for axis in 1:3)) / denominator, 0.0, 1.0)
end

function xi_from_counts(counts, block_counts, sample_counts, edges, minimum;
        angle_counts=nothing, block_sample_counts=nothing,
        parallel_bins=Int[], perpendicular_bins=Int[])
    centers = Float64[]; xi = Float64[]; uncertainty = Float64[]; samples = Int[]
    for bin in axes(counts, 1)
        sample_counts[bin] >= minimum || continue
        parallel, perpendicular = counts[bin, 1], counts[bin, 2]
        denominator = parallel + perpendicular
        denominator > 0 || continue
        analytic_uncertainty = 0.0
        if !isnothing(angle_counts)
            histogram = Float64.(vec(@view angle_counts[bin, :]))
            histogram ./= sum(histogram)
            normalized_parallel = sum(histogram[parallel_bins])
            normalized_perpendicular = sum(histogram[perpendicular_bins])
            normalized_denominator = normalized_parallel + normalized_perpendicular
            parallel_std = length(parallel_bins) > 1 ? std(histogram[parallel_bins]) : 0.0
            perpendicular_std = length(perpendicular_bins) > 1 ?
                std(histogram[perpendicular_bins]) : 0.0
            normalized_denominator > 0 && (analytic_uncertainty = sqrt(4.0 *
                (normalized_perpendicular^2 * parallel_std^2 +
                 normalized_parallel^2 * perpendicular_std^2) /
                normalized_denominator^4))
        end
        block_xi = Float64[]
        for block in axes(block_counts, 2)
            block_parallel = block_counts[bin, block, 1]
            block_perpendicular = block_counts[bin, block, 2]
            block_denominator = block_parallel + block_perpendicular
            block_samples = isnothing(block_sample_counts) ? block_denominator :
                block_sample_counts[bin, block]
            block_samples >= 20 && block_denominator > 0 || continue
            push!(block_xi, (block_parallel - block_perpendicular) / block_denominator)
        end
        push!(centers, sqrt(edges[bin] * edges[bin + 1]))
        push!(xi, (parallel - perpendicular) / denominator)
        block_uncertainty = length(block_xi) >= 3 ? std(block_xi) : NaN
        combined_uncertainty = isfinite(block_uncertainty) ?
            max(analytic_uncertainty, block_uncertainty) : analytic_uncertainty
        push!(uncertainty, combined_uncertainty)
        push!(samples, sample_counts[bin])
    end
    return HROXiCurve(centers, xi, uncertainty, samples)
end

function global_xi(counts, block_counts, sample_counts)
    parallel = sum(@view counts[:, 1]); perpendicular = sum(@view counts[:, 2])
    denominator = parallel + perpendicular
    denominator > 0 || return HROXiGlobal(NaN, NaN, sum(sample_counts))
    block_xi = Float64[]
    for block in axes(block_counts, 2)
        block_parallel = sum(@view block_counts[:, block, 1])
        block_perpendicular = sum(@view block_counts[:, block, 2])
        block_denominator = block_parallel + block_perpendicular
        block_denominator >= 20 || continue
        push!(block_xi, (block_parallel - block_perpendicular) / block_denominator)
    end
    uncertainty = length(block_xi) >= 3 ? std(block_xi) : NaN
    return HROXiGlobal((parallel - perpendicular) / denominator,
        uncertainty, sum(sample_counts))
end

function ibanez_xi_analysis(density, bx, by, bz, vx, vy, vz;
        bins=25, density_range=(1.0e-2, 1.0e3), minimum=1000,
        blocks_per_axis=4, box_size=1.0, axis_order=("x", "y", "z"),
        periodic=nothing, gradient_periodic=false, parallel_cosine=0.75,
        perpendicular_cosine=0.25, cosine_bins=40)
    bins > 1 || error("xi_bins must be greater than one")
    density_min, density_max = Float64.(density_range)
    0 < density_min < density_max || error("xi_density_range must be positive and increasing")
    0 <= perpendicular_cosine < parallel_cosine <= 1 ||
        error("Ibanez cosine thresholds must satisfy 0 <= perpendicular < parallel <= 1")
    cosine_bins > 1 || error("xi_cosine_bins must be greater than one")
    !isnothing(periodic) && (gradient_periodic = Bool(periodic))
    edges = 10 .^ range(log10(density_min), log10(density_max); length=bins + 1)
    cosine_edges = collect(range(0.0, 1.0; length=cosine_bins + 1))
    parallel_bins = [index for index in 1:cosine_bins if
        cosine_edges[index] >= parallel_cosine]
    perpendicular_bins = [index for index in 1:cosine_bins if
        cosine_edges[index + 1] <= perpendicular_cosine]
    nblocks = blocks_per_axis^3
    counts_gradient = zeros(Float64, bins, 2)
    counts_velocity = zeros(Float64, bins, 2)
    blocks_gradient = zeros(Float64, bins, nblocks, 2)
    blocks_velocity = zeros(Float64, bins, nblocks, 2)
    block_samples_gradient = zeros(Int, bins, nblocks)
    block_samples_velocity = zeros(Int, bins, nblocks)
    angles_gradient = zeros(Float64, bins, cosine_bins)
    angles_velocity = zeros(Float64, bins, cosine_bins)
    samples_gradient = zeros(Int, bins); samples_velocity = zeros(Int, bins)
    spacings = grid_spacings(density, box_size; axis_order)
    sigma_v, velocity_mean = velocity_dispersion_1d(density, vx, vy, vz)
    shape = size(density)

    @inbounds for cartesian in CartesianIndices(density)
        condition = Float64(density[cartesian])
        isfinite(condition) && density_min <= condition < density_max || continue
        bin = clamp(searchsortedlast(edges, condition), 1, bins)
        coordinates = Tuple(cartesian)
        magnetic = (Float64(bx[cartesian]), Float64(by[cartesian]), Float64(bz[cartesian]))
        all(isfinite, magnetic) || continue
        gradient = Float64.(gradient_at(density, coordinates..., spacings, gradient_periodic))
        velocity = (Float64(vx[cartesian]) - velocity_mean[1],
            Float64(vy[cartesian]) - velocity_mean[2],
            Float64(vz[cartesian]) - velocity_mean[3])
        block = xi_block_index(coordinates, shape, blocks_per_axis)
        gradient_cosine = absolute_cosine(gradient, magnetic)
        if isfinite(gradient_cosine)
            samples_gradient[bin] += 1
            block_samples_gradient[bin, block] += 1
            cosine_bin = clamp(searchsortedlast(cosine_edges, gradient_cosine), 1, cosine_bins)
            angles_gradient[bin, cosine_bin] += 1
            category = xi_category(gradient_cosine, parallel_cosine, perpendicular_cosine)
            if category > 0
                counts_gradient[bin, category] += 1
                blocks_gradient[bin, block, category] += 1
            end
        end
        velocity_cosine = absolute_cosine(velocity, magnetic)
        if isfinite(velocity_cosine)
            samples_velocity[bin] += 1
            block_samples_velocity[bin, block] += 1
            cosine_bin = clamp(searchsortedlast(cosine_edges, velocity_cosine), 1, cosine_bins)
            angles_velocity[bin, cosine_bin] += 1
            category = xi_category(velocity_cosine, parallel_cosine, perpendicular_cosine)
            if category > 0
                counts_velocity[bin, category] += 1
                blocks_velocity[bin, block, category] += 1
            end
        end
    end

    gradient_curve = xi_from_counts(counts_gradient, blocks_gradient,
        samples_gradient, edges, minimum; angle_counts=angles_gradient,
        block_sample_counts=block_samples_gradient, parallel_bins, perpendicular_bins)
    velocity_curve = xi_from_counts(counts_velocity, blocks_velocity,
        samples_velocity, edges, minimum; angle_counts=angles_velocity,
        block_sample_counts=block_samples_velocity, parallel_bins, perpendicular_bins)
    return (gradient=gradient_curve, velocity=velocity_curve,
        gradient_global=global_xi(counts_gradient, blocks_gradient, samples_gradient),
        velocity_global=global_xi(counts_velocity, blocks_velocity, samples_velocity),
        sigma_v=sigma_v, edges=collect(edges))
end

function xi_plot_limits(curves, fallback)
    centers = reduce(vcat, (curve.centers for curve in curves); init=Float64[])
    isempty(centers) && return fallback
    return (10.0^floor(log10(minimum(centers))),
        10.0^ceil(log10(maximum(centers))))
end

function add_xi_background!(axis, xlimits)
    x = collect(xlimits)
    band!(axis, x, zeros(2), ones(2); color=(PLOT_BLUE, 0.13))
    band!(axis, x, -ones(2), zeros(2); color=(PLOT_ORANGE, 0.13))
    hlines!(axis, [0.0]; color=PLOT_MUTED, linewidth=1.5, linestyle=:dash)
end

function set_xi_log_ticks!(axis, xlimits)
    first_exponent = ceil(Int, log10(first(xlimits)))
    last_exponent = floor(Int, log10(last(xlimits)))
    exponents = collect(first_exponent:last_exponent)
    axis.xtickformat = values -> [latexstring("10^{", round(Int, log10(value)), "}")
        for value in values]
    axis.xticks = 10.0 .^ exponents
end

function plot_xi_curve!(axis, curve::HROXiCurve, xlimits)
    add_xi_background!(axis, xlimits)
    valid = findall(isfinite, curve.xi)
    uncertainty_valid = filter(index -> isfinite(curve.uncertainty[index]), valid)
    !isempty(uncertainty_valid) && errorbars!(axis, curve.centers[uncertainty_valid],
        curve.xi[uncertainty_valid], curve.uncertainty[uncertainty_valid];
        color=(PLOT_INK, 0.72), whiskerwidth=8, linewidth=1.3)
    if !isempty(valid)
        lines!(axis, curve.centers[valid], curve.xi[valid]; color=PLOT_INK, linewidth=2.7)
        scatter!(axis, curve.centers[valid], curve.xi[valid]; color=:white,
            strokecolor=PLOT_INK, strokewidth=1.5, markersize=9)
    end
    xlims!(axis, xlimits...); ylims!(axis, -1.0, 1.0)
    set_xi_log_ticks!(axis, xlimits)
    axis.xgridvisible = false; axis.ygridvisible = false
    axis.xminorticksvisible = true; axis.yminorticksvisible = true
end

function plot_ibanez_xi!(files, fields, metadata, cfg, spec, output_dir, formats, overwrite)
    Bool(get(spec, "xi_plot", false)) || return nothing
    velocity_names = string.(get(spec, "velocity_vector", String[]))
    length(velocity_names) == 3 || error("xi_plot requires velocity_vector with three fields")
    scalar_name = string(spec["scalar"]); magnetic_names = string.(spec["vector"])
    required = [scalar_name; magnetic_names; velocity_names]
    all(haskey(fields, name) for name in required) || error("xi_plot references missing fields")
    arrays = [fields[name] for name in required]
    enforce_working_set(cfg, length(arrays) * sizeof(Float32) * length(first(arrays));
        context="Ibanez xi plot")
    grid = get(cfg, "grid", Dict())
    result = ibanez_xi_analysis(arrays...;
        bins=Int(get(spec, "xi_bins", 25)),
        density_range=Tuple(Float64.(get(spec, "xi_density_range", [1.0e-2, 1.0e3]))),
        minimum=Int(get(spec, "xi_minimum_per_bin", 1000)),
        blocks_per_axis=Int(get(spec, "xi_blocks_per_axis", 4)),
        box_size=get(grid, "box_size", 1.0),
        axis_order=string.(get(grid, "axis_order", ["x", "y", "z"])),
        gradient_periodic=Bool(get(spec, "xi_gradient_periodic", false)),
        parallel_cosine=Float64(get(spec, "xi_parallel_cosine", 0.75)),
        perpendicular_cosine=Float64(get(spec, "xi_perpendicular_cosine", 0.25)),
        cosine_bins=Int(get(spec, "xi_cosine_bins", 40)))
    xlimits = xi_plot_limits((result.gradient, result.velocity),
        (first(result.edges), last(result.edges)))
    condition_label = field_label(string(get(spec, "condition", scalar_name)), metadata[scalar_name])
    figure = publication_figure(size=(1160, 480))
    gradient_axis = latex_axis(figure[1, 1]; xlabel=condition_label,
        ylabel=latexstring("\\xi_{\\nabla n,B}"), xscale=log10)
    velocity_axis = latex_axis(figure[1, 2]; xlabel=condition_label,
        ylabel=latexstring("\\xi_{\\delta v,B}"), xscale=log10)
    plot_xi_curve!(gradient_axis, result.gradient, xlimits)
    plot_xi_curve!(velocity_axis, result.velocity, xlimits)
    colgap!(figure.layout, 32)
    name = string(spec["name"])
    save_figure!(files, figure, output_dir, "xi_$(sanitize(name))", formats; overwrite)
    return result
end

function plot_xi_vs_sigma_v!(files, simulation_dirs, output_dir, formats;
        overwrite=false, box_size=(50.0, 50.0, 50.0), axis_order=("x", "y", "z"))
    pending_formats = overwrite ? collect(formats) : filter(formats) do format
        !isfile(joinpath(output_dir, "xi_vs_sigma_v.$(lowercase(string(format)))"))
    end
    if isempty(pending_formats)
        @info "Global xi versus velocity-dispersion figure already exists; skipping" output_dir
        return files
    end
    sigma_values = Float64[]; gradient_values = Float64[]; velocity_values = Float64[]
    gradient_errors = Float64[]; velocity_errors = Float64[]
    for directory in simulation_dirs
        density = read_fits_cube(joinpath(directory, "density.fits"))
        bx = read_fits_cube(joinpath(directory, "Bx.fits")); by = read_fits_cube(joinpath(directory, "By.fits"))
        bz = read_fits_cube(joinpath(directory, "Bz.fits")); vx = read_fits_cube(joinpath(directory, "Vx.fits"))
        vy = read_fits_cube(joinpath(directory, "Vy.fits")); vz = read_fits_cube(joinpath(directory, "Vz.fits"))
        result = ibanez_xi_analysis(density, bx, by, bz, vx, vy, vz;
            box_size, axis_order, minimum=1000)
        push!(sigma_values, result.sigma_v)
        push!(gradient_values, result.gradient_global.xi)
        push!(velocity_values, result.velocity_global.xi)
        push!(gradient_errors, result.gradient_global.uncertainty)
        push!(velocity_errors, result.velocity_global.uncertainty)
        GC.gc(true)
    end
    isempty(sigma_values) && error("No simulations available for xi versus velocity dispersion")
    xmin, xmax = extrema(sigma_values)
    xmin == xmax && (xmin *= 0.9; xmax *= 1.1)
    xlimits = (0.85xmin, 1.15xmax)
    figure = publication_figure()
    axis = latex_axis(figure[1, 1];
        xlabel=latexstring("\\sigma_v\\ [\\mathrm{km\\,s^{-1}}]"),
        ylabel=latexstring("\\xi"), xscale=log10)
    add_xi_background!(axis, xlimits)
    valid_gradient = findall(index -> isfinite(gradient_values[index]) &&
        isfinite(gradient_errors[index]), eachindex(gradient_values))
    valid_velocity = findall(index -> isfinite(velocity_values[index]) &&
        isfinite(velocity_errors[index]), eachindex(velocity_values))
    errorbars!(axis, sigma_values[valid_gradient], gradient_values[valid_gradient],
        gradient_errors[valid_gradient]; color=(PLOT_INK, 0.72), whiskerwidth=8,
        linewidth=1.3)
    scatter!(axis, sigma_values[valid_gradient], gradient_values[valid_gradient];
        color=:white, strokecolor=PLOT_INK, strokewidth=1.7,
        marker=:circle, markersize=11,
        label=latexstring("\\xi_{\\nabla n,B}"))
    errorbars!(axis, sigma_values[valid_velocity], velocity_values[valid_velocity],
        velocity_errors[valid_velocity]; color=(PLOT_BLUE, 0.72), whiskerwidth=8,
        linewidth=1.3)
    scatter!(axis, sigma_values[valid_velocity], velocity_values[valid_velocity];
        color=:white, strokecolor=PLOT_BLUE, strokewidth=1.7,
        marker=:utriangle, markersize=12,
        label=latexstring("\\xi_{\\delta v,B}"))
    xlims!(axis, xlimits...); ylims!(axis, -1.0, 1.0)
    axis.xgridvisible = false; axis.ygridvisible = false
    publication_legend!(axis; position=:rb)
    save_figure!(files, figure, output_dir, "xi_vs_sigma_v", pending_formats; overwrite)
    return files
end

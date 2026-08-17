"""Logarithmically spaced lags in cells, up to `minimum(shape) / divisor`."""
default_structure_lags(shape, divisor, count) = unique!(round.(Int,
    10 .^ range(0, log10(max(1, minimum(shape) ÷ divisor)); length=count)))

"""
Draw a random cell and the cell displaced by `lag` along a random axis.

Returns `(inside, i, j, k, ti, tj, tk)`; `inside` is false when the partner
falls outside a non-periodic box. Keeping the coordinates in scalars rather
than a vector avoids one heap allocation per sample, which dominates the cost
of the structure-function loops.
"""
@inline function random_increment_pair(rng, shape, lag, periodic)
    axis = rand(rng, 1:3)
    direction = rand(rng, Bool) ? lag : -lag
    i = rand(rng, 1:shape[1]); j = rand(rng, 1:shape[2]); k = rand(rng, 1:shape[3])
    target = (i, j, k)[axis] + direction
    if periodic
        target = mod1(target, shape[axis])
    elseif !(1 <= target <= shape[axis])
        return (false, i, j, k, i, j, k)
    end
    return (true, i, j, k, axis == 1 ? target : i, axis == 2 ? target : j,
        axis == 3 ? target : k)
end

"""
Absolute structure functions `S_p(l) = <|f(x+l) - f(x)|^p>` estimated by Monte
Carlo sampling.

Each lag is seeded independently from `seed`, so results are reproducible and
independent of the number of threads.
"""
function structure_functions(cube; orders=1:3, lags=nothing, samples_per_lag=100_000,
        seed=1234, periodic=true)
    selected_lags = isnothing(lags) ?
        default_structure_lags(size(cube), 2, 18) : Int.(lags)
    all(>(0), selected_lags) || error("Structure-function lags must be positive")
    selected_orders = Int.(orders)
    shape = size(cube)
    values = fill(NaN, length(selected_lags), length(selected_orders))
    counts = zeros(Int, length(selected_lags))
    Threads.@threads for lag_index in eachindex(selected_lags)
        lag = selected_lags[lag_index]
        rng = MersenneTwister(seed + lag_index)
        sums = zeros(Float64, length(selected_orders))
        count = 0
        @inbounds for _ in 1:samples_per_lag
            inside, i, j, k, ti, tj, tk =
                random_increment_pair(rng, shape, lag, periodic)
            inside || continue
            first = Float64(cube[i, j, k])
            second = Float64(cube[ti, tj, tk])
            isfinite(first) && isfinite(second) || continue
            delta = abs(second - first)
            for (order_index, order) in enumerate(selected_orders)
                sums[order_index] += delta^order
            end
            count += 1
        end
        counts[lag_index] = count
        count > 0 && (values[lag_index, :] .= sums ./ count)
    end
    return selected_lags, selected_orders, values, counts
end

function structure_plot_scales(cube, lags, box_size, axis_order;
        injection_modes=4.0, dissipation_cells=4.0, box_unit="L")
    spacings = collect(grid_spacings(cube, box_size; axis_order))
    isotropic = maximum(spacings) / minimum(spacings) <= 1 + 1e-8
    if isotropic
        separation = Float64.(lags) .* mean(spacings)
        injection_scale = 2π / injection_wavenumber(box_size, axis_order;
            modes=injection_modes)
        dissipation_scale = 2π / dissipation_wavenumber(size(cube), box_size,
            axis_order; cells=dissipation_cells)
        label = latexstring("\\ell\\ [\\mathrm{", box_unit, "]}")
    else
        separation = Float64.(lags)
        injection_scale = minimum(size(cube)) / Float64(injection_modes)
        dissipation_scale = Float64(dissipation_cells)
        label = latexstring("\\ell\\ [\\mathrm{cells}]")
    end
    return separation, dissipation_scale, injection_scale, label
end

function add_structure_ranges!(axis, separation, dissipation_scale, injection_scale)
    xmin, xmax = extrema(separation)
    if xmin < dissipation_scale
        vspan!(axis, xmin, min(dissipation_scale, xmax); color=(PLOT_MUTED, 0.16))
    end
    if injection_scale < xmax
        vspan!(axis, max(injection_scale, xmin), xmax; color=(PLOT_GOLD, 0.18))
    end
    xmin <= dissipation_scale <= xmax && vlines!(axis, [dissipation_scale];
        color=PLOT_MUTED, linestyle=:dashdot, linewidth=1.8)
    xmin <= injection_scale <= xmax && vlines!(axis, [injection_scale];
        color=PLOT_ORANGE, linestyle=:dashdot, linewidth=1.8)
    return axis
end

function add_structure_fit!(axis, separation, values, order, fit_range)
    slope, slope_std, points = spectral_fit(separation, values, fit_range)
    isfinite(slope) || return (slope, slope_std, points)
    selected = findall(index -> fit_range[1] <= separation[index] <= fit_range[2] &&
        isfinite(values[index]) && values[index] > 0, eachindex(separation))
    anchor = selected[cld(length(selected), 2)]
    x = Float64.(separation[selected])
    y = Float64(values[anchor]) .* (x ./ separation[anchor]) .^ slope
    uncertainty = isfinite(slope_std) ? @sprintf("%.2f", slope_std) : "\\mathrm{nan}"
    lines!(axis, x, y; color=PLOT_INK, linestyle=:dashdot, linewidth=2.2,
        label=latexstring("\\zeta_{", order, "}=", @sprintf("%.2f", slope),
            "\\pm", uncertainty))
    return slope, slope_std, points
end

kolmogorov_ess(order::Real) = Float64(order) / 3
she_leveque_ess(order::Real) = Float64(order) / 9 +
    2 * (1 - (2 / 3)^(Float64(order) / 3))
boldyrev_ess(order::Real) = Float64(order) / 9 +
    1 - (1 / 3)^(Float64(order) / 3)

function default_thermal_phases()
    return [
        (name="CNM", minimum=0.0, maximum=300.0),
        (name="UNM", minimum=300.0, maximum=5000.0),
        (name="WNM", minimum=5000.0, maximum=Inf),
    ]
end

function configured_thermal_phases(settings)
    raw = get(settings, "phases", nothing)
    isnothing(raw) && return default_thermal_phases()
    return [(name=string(phase["name"]),
        minimum=Float64(get(phase, "temperature_min", 0.0)),
        maximum=Float64(get(phase, "temperature_max", Inf))) for phase in raw]
end

function phase_velocity_structure(vx, vy, vz, temperature, phases; orders=1:5,
        lags=nothing, samples_per_lag=100_000, seed=1234, periodic=true)
    shape = size(vx)
    all(size(field) == shape for field in (vy, vz, temperature)) ||
        error("Phase ESS fields must have identical sizes")
    selected_lags = isnothing(lags) ?
        default_structure_lags(shape, 2, 18) : Int.(lags)
    selected_orders = Int.(orders)
    sums = zeros(Float64, length(selected_lags), length(phases), length(selected_orders))
    counts = zeros(Int, length(selected_lags), length(phases))
    Threads.@threads for lag_index in eachindex(selected_lags)
        lag = selected_lags[lag_index]
        rng = MersenneTwister(seed + lag_index)
        @inbounds for _ in 1:samples_per_lag
            inside, i, j, k, ti, tj, tk =
                random_increment_pair(rng, shape, lag, periodic)
            inside || continue
            first_temperature = Float64(temperature[i, j, k])
            second_temperature = Float64(temperature[ti, tj, tk])
            isfinite(first_temperature) && isfinite(second_temperature) || continue
            phase_index = findfirst(
                phase -> phase.minimum <= first_temperature < phase.maximum &&
                    phase.minimum <= second_temperature < phase.maximum, phases)
            isnothing(phase_index) && continue
            du2 = 0.0
            for component in (vx, vy, vz)
                first_value = Float64(component[i, j, k])
                second_value = Float64(component[ti, tj, tk])
                isfinite(first_value) && isfinite(second_value) || (du2 = NaN; break)
                du2 += (second_value - first_value)^2
            end
            isfinite(du2) || continue
            increment = sqrt(du2)
            for (order_index, order) in enumerate(selected_orders)
                sums[lag_index, phase_index, order_index] += increment^order
            end
            counts[lag_index, phase_index] += 1
        end
    end
    values = fill(NaN, size(sums))
    for lag_index in eachindex(selected_lags), phase_index in eachindex(phases)
        count = counts[lag_index, phase_index]
        count > 0 && (values[lag_index, phase_index, :] .=
            sums[lag_index, phase_index, :] ./ count)
    end
    return selected_lags, selected_orders, values, counts
end

function ess_exponents(structure, orders, selected_lags, lag_range; minimum_samples=nothing,
        bootstrap_replicates=0, seed=1234)
    third = findfirst(==(3), orders)
    isnothing(third) && error("ESS requires third-order structure functions")
    exponents = fill(NaN, length(orders)); uncertainties = similar(exponents)
    points = zeros(Int, length(orders))
    for (order_index, _) in enumerate(orders)
        valid = findall(index -> lag_range[1] <= selected_lags[index] <= lag_range[2] &&
            isfinite(structure[index, third]) && structure[index, third] > 0 &&
            isfinite(structure[index, order_index]) && structure[index, order_index] > 0,
            eachindex(selected_lags))
        if !isnothing(minimum_samples)
            valid = filter(index -> minimum_samples[index] > 0, valid)
        end
        points[order_index] = length(valid)
        length(valid) >= 3 || continue
        x = log10.(structure[valid, third]); y = log10.(structure[valid, order_index])
        # Lags are sampled independently, so a lag averaging more increments
        # constrains the exponent better; weight by sqrt of the sample count.
        w = isnothing(minimum_samples) ? ones(length(valid)) :
            sqrt.(Float64.(minimum_samples[valid]))
        total = sum(w)
        total > 0 || continue
        xmean=sum(w .* x)/total; ymean=sum(w .* y)/total
        denominator=sum(w .* abs2.(x .- xmean))
        denominator > 0 || continue
        slope=sum(w .* (x .- xmean) .* (y .- ymean)) / denominator
        residual=y .- (ymean - slope*xmean .+ slope .* x)
        exponents[order_index]=slope
        uncertainties[order_index]=sqrt(sum(w .* abs2.(residual)) /
            (length(valid)-2) / denominator)
        if bootstrap_replicates > 1
            rng=MersenneTwister(seed+order_index); bootstrap=Float64[]
            for _ in 1:bootstrap_replicates
                selected=rand(rng,eachindex(x),length(x)); bx=x[selected]; by=y[selected]
                bxmean=mean(bx); bymean=mean(by); bden=sum(abs2,bx.-bxmean)
                bden>0 && push!(bootstrap,sum((bx.-bxmean).*(by.-bymean))/bden)
            end
            length(bootstrap)>1 && (uncertainties[order_index]=std(bootstrap;corrected=false))
        end
    end
    return exponents, uncertainties, points
end

function plot_phase_ess!(files, fields, cfg, settings, output_dir, formats, overwrite,
        injection_modes, dissipation_cells)
    Bool(get(settings, "phase_ess", true)) || return files
    velocity_names = string.(get(settings, "phase_ess_velocity", ["Vx", "Vy", "Vz"]))
    temperature_name = string(get(settings, "phase_ess_temperature", "temperature"))
    all(haskey(fields, name) for name in [velocity_names; temperature_name]) ||
        error("Phase ESS references missing velocity or temperature fields")
    phases = configured_thermal_phases(settings)
    cube = fields[first(velocity_names)]
    enforce_working_set(cfg, 4 * sizeof(Float32) * length(cube);
        context="phase-conditioned ESS")
    lags, orders, values, counts = phase_velocity_structure(
        (fields[name] for name in velocity_names)..., fields[temperature_name], phases;
        orders=Int.(get(settings, "phase_ess_orders", [1, 2, 3, 4, 5])),
        samples_per_lag=Int(get(settings, "phase_ess_samples", 100_000)),
        seed=Int(get(settings, "seed", 1234)),
        periodic=Bool(get(get(cfg, "grid", Dict()), "periodic", true)))
    lag_range = [dissipation_cells, minimum(size(cube)) / injection_modes]
    fig = publication_figure(size=(900, 620)); axis = latex_axis(fig[1, 1],
        xlabel=latexstring("p"), ylabel=latexstring("\\zeta_p/\\zeta_3\\ (\\mathrm{ESS})"))
    phase_results = NamedTuple[]
    for (phase_index, phase) in enumerate(phases)
        exponents, errors, points = ess_exponents(values[:, phase_index, :], orders,
            lags, lag_range; minimum_samples=counts[:, phase_index],
            bootstrap_replicates=Int(get(settings,"phase_ess_bootstrap_replicates",200)),
            seed=Int(get(settings,"seed",1234))+phase_index)
        valid = findall(isfinite, exponents)
        isempty(valid) && continue
        color=series_color(phase_index)
        lines!(axis, orders[valid], exponents[valid]; color, linewidth=2.7,
            label=latex_legend_label(phase.name))
        scatter!(axis, orders[valid], exponents[valid]; color=:white,
            strokecolor=color, strokewidth=1.5, markersize=10,
            marker=series_marker(phase_index))
        errorbars!(axis, orders[valid], exponents[valid], errors[valid];
            color=(color, 0.72), whiskerwidth=7, linewidth=1.2)
        push!(phase_results, (; phase, exponents, errors, points))
    end
    order_values = Float64.(orders)
    lines!(axis, order_values, kolmogorov_ess.(order_values); color=PLOT_INK,
        linestyle=:dash, linewidth=2.0, label=latex_legend_label("Kolmogorov"))
    lines!(axis, order_values, she_leveque_ess.(order_values); color=PLOT_MUTED,
        linestyle=:dot, linewidth=2.2, label=latex_legend_label("She-Leveque"))
    lines!(axis, order_values, boldyrev_ess.(order_values); color=PLOT_INK,
        linestyle=:dashdot, linewidth=2.0, label=latex_legend_label("Boldyrev"))
    axis.xticks = orders
    publication_legend!(axis; position=:lt)
    save_figure!(files, fig, output_dir, "phase_structure_ess", formats; overwrite)
    if save_data(cfg)
        path=joinpath(output_dir, "phase_structure_ess.csv")
        write_text_output(path; overwrite) do io
            println(io, "phase,order,zeta_over_zeta3,uncertainty,fit_points")
            for result in phase_results, index in eachindex(orders)
                @printf(io, "%s,%d,%.8e,%.8e,%d\n", result.phase.name, orders[index],
                    result.exponents[index], result.errors[index], result.points[index])
            end
        end
        push!(files,path)
    end
    return files
end

function radial_autocorrelation(cube; bins=60, box_size=1.0, axis_order=("x", "y", "z"))
    A = Float32.(cube)
    finite_values = filter(isfinite, A)
    isempty(finite_values) && error("Autocorrelation requires finite values")
    replacement = mean(finite_values)
    @inbounds for i in eachindex(A)
        isfinite(A[i]) || (A[i] = replacement)
    end
    A .-= mean(A)
    transformed = rfft(A)
    correlation = irfft(abs2.(transformed), size(A, 1))
    correlation ./= correlation[1, 1, 1]
    lengths = box_lengths(box_size, axis_order)
    spacing = ntuple(d -> lengths[d] / size(A, d), 3)
    rmax = minimum(lengths) / 2
    edges = collect(range(0, rmax; length=bins + 1))
    sums = zeros(Float64, bins); counts = zeros(Int, bins)
    @inbounds for k in axes(A, 3), j in axes(A, 2), i in axes(A, 1)
        offsets = (min(i - 1, size(A, 1) - i + 1) * spacing[1],
            min(j - 1, size(A, 2) - j + 1) * spacing[2],
            min(k - 1, size(A, 3) - k + 1) * spacing[3])
        radius = sqrt(sum(abs2, offsets))
        radius <= rmax || continue
        bin = clamp(searchsortedlast(edges, radius), 1, bins)
        sums[bin] += correlation[i, j, k]; counts[bin] += 1
    end
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    keep = counts .> 0
    values = sums[keep] ./ counts[keep]
    crossing = findfirst(<=(exp(-1)), values)
    correlation_length = isnothing(crossing) ? NaN : centers[keep][crossing]
    return centers[keep], values, counts[keep], correlation_length
end

function plot_advanced_diagnostics!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "diagnostics", Dict())
    periodic = Bool(get(get(cfg, "grid", Dict()), "periodic", true))
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    box_unit = string(get(grid, "box_unit", "L"))
    spectra = get(cfg, "spectra", Dict())
    injection_modes = Float64(get(settings, "structure_injection_modes",
        get(spectra, "injection_modes", 4.0)))
    dissipation_cells = Float64(get(settings, "structure_dissipation_cells",
        get(spectra, "dissipation_cells", 4.0)))
    for name in string.(get(settings, "structure_fields", String[]))
        haskey(fields, name) || error("Structure function references missing field '$name'")
        cube = fields[name]
        lags, orders, values, counts = structure_functions(cube;
            orders=Int.(get(settings, "structure_orders", [1, 2, 3])),
            samples_per_lag=Int(get(settings, "structure_samples", 100_000)),
            seed=Int(get(settings, "seed", 1234)), periodic)
        separation, dissipation_scale, injection_scale, separation_label =
            structure_plot_scales(cube, lags, box_size, axis_order;
                injection_modes, dissipation_cells, box_unit)
        physical_fit_range = [dissipation_scale, injection_scale]
        if save_data(cfg)
            path = joinpath(output_dir, "structure_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, join(["lag_cells", "separation", "samples",
                    ["S$order" for order in orders]...], ','))
                for i in eachindex(lags)
                    println(io, join([lags[i], separation[i], counts[i], values[i, :]...], ','))
                end
            end
            push!(files, path)
        end
        panel_width = 410
        fig = publication_figure(size=(max(900, panel_width * length(orders)), 520))
        for (column, order) in enumerate(orders)
            axis = latex_axis(fig[1, column]; xscale=log10, yscale=log10,
                xlabel=separation_label,
                ylabel=latexstring("S_{", order, "}(\\ell)"))
            add_structure_ranges!(axis, separation, dissipation_scale, injection_scale)
            valid = findall(index -> isfinite(values[index, column]) &&
                values[index, column] > 0, eachindex(separation))
            if !isempty(valid)
                color = series_color(column)
                lines!(axis, separation[valid], values[valid, column]; color, linewidth=2.7)
                scatter!(axis, separation[valid], values[valid, column]; color=:white,
                    strokecolor=color, strokewidth=1.3, markersize=7,
                    marker=series_marker(column))
                fit_range = selected_fit_range(separation, values[:, column],
                    physical_fit_range, settings)
                slope, _, _ = add_structure_fit!(axis, separation, values[:, column],
                    order, fit_range)
                isfinite(slope) && publication_legend!(axis; position=:lt)
            end
        end
        colgap!(fig.layout, 34)
        save_figure!(files, fig, output_dir, "structure_$(sanitize(name))", formats;
            overwrite)
    end
    plot_phase_ess!(files, fields, cfg, settings, output_dir, formats, overwrite,
        injection_modes, dissipation_cells)
    save_data(cfg) || return files
    for name in string.(get(settings, "correlation_fields", String[]))
        haskey(fields, name) || error("Correlation references missing field '$name'")
        radius, correlation, modes, length_scale = radial_autocorrelation(fields[name];
            bins=Int(get(settings, "correlation_bins", 60)), box_size, axis_order)
        path = joinpath(output_dir, "correlation_$(sanitize(name)).csv")
        write_text_output(path; overwrite) do io
            println(io, "separation,autocorrelation,modes")
            for i in eachindex(radius)
                @printf(io, "%.8e,%.8e,%d\n", radius[i], correlation[i], modes[i])
            end
        end
        push!(files, path)
        scale_path = joinpath(output_dir, "correlation_$(sanitize(name))_scale.toml")
        write_text_output(scale_path; overwrite) do io
            TOML.print(io, Dict("e_folding_length" => length_scale); sorted=true)
        end
        push!(files, scale_path)
    end
    return files
end

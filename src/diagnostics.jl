function periodic_index(index, offset, length)
    return mod1(index + offset, length)
end

function structure_functions(cube; orders=1:3, lags=nothing, samples_per_lag=100_000,
        seed=1234, periodic=true)
    selected_lags = isnothing(lags) ? unique!(round.(Int,
        10 .^ range(0, log10(max(1, minimum(size(cube)) ÷ 2)); length=18))) : Int.(lags)
    all(>(0), selected_lags) || error("Structure-function lags must be positive")
    selected_orders = Int.(orders)
    rng = MersenneTwister(seed)
    values = fill(NaN, length(selected_lags), length(selected_orders))
    counts = zeros(Int, length(selected_lags))
    for (li, lag) in enumerate(selected_lags)
        sums = zeros(Float64, length(selected_orders))
        for _ in 1:samples_per_lag
            axis = rand(rng, 1:3); direction = rand(rng, Bool) ? lag : -lag
            i = rand(rng, axes(cube, 1)); j = rand(rng, axes(cube, 2)); k = rand(rng, axes(cube, 3))
            coordinates = [i, j, k]
            target = coordinates[axis] + direction
            if periodic
                coordinates[axis] = mod1(target, size(cube, axis))
            elseif !(1 <= target <= size(cube, axis))
                continue
            else
                coordinates[axis] = target
            end
            first = Float64(cube[i, j, k])
            second = Float64(cube[coordinates...])
            isfinite(first) && isfinite(second) || continue
            delta = abs(second - first)
            @inbounds for (oi, order) in enumerate(selected_orders)
                sums[oi] += delta^order
            end
            counts[li] += 1
        end
        counts[li] > 0 && (values[li, :] .= sums ./ counts[li])
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

function physics_diagnostics(fields, spec; stride=1)
    required = Dict(
        "density" => string(get(spec, "density", "density")),
        "temperature" => string(get(spec, "temperature", "temperature")),
    )
    velocity = string.(get(spec, "velocity", ["Vx", "Vy", "Vz"]))
    magnetic = string.(get(spec, "magnetic", ["Bx", "By", "Bz"]))
    names = [collect(values(required)); velocity; magnetic]
    all(haskey(fields, name) for name in names) ||
        error("Physics diagnostics reference missing fields")
    gamma = Float64(get(spec, "gamma", 5 / 3))
    mu = Float64(get(spec, "mean_molecular_weight", 1.27))
    velocity_to_cms = Float64(get(spec, "velocity_to_cms", 1e5))
    magnetic_to_gauss = Float64(get(spec, "magnetic_to_gauss", 1e-3))
    k_B = 1.380649e-16; m_H = 1.6735575e-24
    mach_sonic = Float64[]; mach_alfven = Float64[]; beta = Float64[]
    density = fields[required["density"]]; temperature = fields[required["temperature"]]
    vx, vy, vz = (fields[name] for name in velocity)
    bx, by, bz = (fields[name] for name in magnetic)
    for index in 1:stride:length(density)
        n = Float64(density[index]); T = Float64(temperature[index])
        v = velocity_to_cms * sqrt(Float64(vx[index])^2 + Float64(vy[index])^2 + Float64(vz[index])^2)
        B = magnetic_to_gauss * sqrt(Float64(bx[index])^2 + Float64(by[index])^2 + Float64(bz[index])^2)
        isfinite(n + T + v + B) && n > 0 && T > 0 && B > 0 || continue
        rho = mu * m_H * n
        sound = sqrt(gamma * k_B * T / (mu * m_H))
        alfven = B / sqrt(4π * rho)
        push!(mach_sonic, v / sound)
        push!(mach_alfven, v / alfven)
        push!(beta, 8π * n * k_B * T / B^2)
    end
    return Dict("mach_sonic" => mach_sonic, "mach_alfven" => mach_alfven, "plasma_beta" => beta)
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
        lags, orders, values, counts = structure_functions(fields[name];
            orders=Int.(get(settings, "structure_orders", [1, 2, 3])),
            samples_per_lag=Int(get(settings, "structure_samples", 100_000)),
            seed=Int(get(settings, "seed", 1234)), periodic)
        separation, dissipation_scale, injection_scale, separation_label =
            structure_plot_scales(cube, lags, box_size, axis_order;
                injection_modes, dissipation_cells, box_unit)
        fit_range = [dissipation_scale, injection_scale]
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
                slope, _, _ = add_structure_fit!(axis, separation, values[:, column],
                    order, fit_range)
                isfinite(slope) && publication_legend!(axis; position=:lt)
            end
        end
        colgap!(fig.layout, 34)
        save_figure!(files, fig, output_dir, "structure_$(sanitize(name))", formats;
            overwrite)
    end
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
    if Bool(get(settings, "physics", false))
        diagnostics = physics_diagnostics(fields, settings;
            stride=Int(get(cfg["run"], "sample_stride", 1)))
        path = joinpath(output_dir, "physics_diagnostics.csv")
        write_text_output(path; overwrite) do io
            println(io, "quantity,samples,median,mean,std,q16,q84")
            for name in sort!(collect(keys(diagnostics)))
                values = diagnostics[name]
                isempty(values) && continue
                @printf(io, "%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e\n", name, length(values),
                    median(values), mean(values), std(values; corrected=false),
                    quantile(values, 0.16), quantile(values, 0.84))
            end
        end
        push!(files, path)
    end
    return files
end

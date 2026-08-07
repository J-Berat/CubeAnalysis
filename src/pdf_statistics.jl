# Density and column-density probability distributions: the lognormal fit and
# its driving parameter b, the thermal-phase budget, and the power-law tail of
# the column-density PDF.

"""
Logarithmic density contrast `s = ln(n / <n>)` over the strided cells.

`<n>` is the volume-weighted mean, which is the normalisation the lognormal
prediction for supersonic turbulence is written in.
"""
function log_density_contrast(density; stride=1)
    values = Float64[]
    sizehint!(values, cld(length(density), stride))
    @inbounds for index in 1:stride:length(density)
        n = Float64(density[index])
        isfinite(n) && n > 0 || continue
        push!(values, n)
    end
    isempty(values) && error("The density PDF needs positive finite values")
    mean_density = mean(values)
    contrast = log.(values ./ mean_density)
    return (contrast=contrast, mean_density=mean_density,
        sigma=std(contrast; corrected=false), mean_contrast=mean(contrast),
        samples=length(contrast))
end

"""
Turbulence driving parameter from `sigma_s^2 = ln(1 + b^2 M^2)`.

`b ~ 1/3` corresponds to purely solenoidal forcing and `b ~ 1` to purely
compressive forcing.
"""
driving_parameter(sigma, mach) =
    mach > 0 && isfinite(sigma) ? sqrt(expm1(sigma^2)) / mach : NaN

"""
Driving parameter corrected for magnetic pressure, from
`sigma_s^2 = ln(1 + b^2 M^2 beta / (beta + 1))`.
"""
function magnetised_driving_parameter(sigma, mach, beta)
    mach > 0 && beta > 0 && isfinite(sigma) && isfinite(beta) || return NaN
    return sqrt(expm1(sigma^2) * (beta + 1) / beta) / mach
end

"""Gaussian density with the lognormal constraint `mu = -sigma^2 / 2`."""
lognormal_density(s, sigma) =
    exp(-(s + sigma^2 / 2)^2 / (2 * sigma^2)) / sqrt(2π * sigma^2)

gaussian_density(s, mean_value, sigma) =
    exp(-(s - mean_value)^2 / (2 * sigma^2)) / sqrt(2π * sigma^2)

function write_density_pdf!(files, statistics, mach, beta, b_value, b_magnetised,
        output_dir; overwrite=true)
    path = joinpath(output_dir, "density_pdf_fit.toml")
    write_text_output(path; overwrite) do io
        TOML.print(io, Dict(
            "sigma_s" => statistics.sigma,
            "mean_s" => statistics.mean_contrast,
            "lognormal_mean_s" => -statistics.sigma^2 / 2,
            "mean_density" => statistics.mean_density,
            "samples" => statistics.samples,
            "mach_sonic" => mach,
            "plasma_beta" => beta,
            "b" => b_value,
            "b_magnetised" => b_magnetised); sorted=true)
    end
    push!(files, path)
    return path
end

function plot_density_pdf!(files, fields, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "diagnostics", Dict())
    density_name = string(get(settings, "density", "density"))
    if !haskey(fields, density_name)
        @info "Skipping the density PDF; the cube lacks a density field" density_name
        return files
    end
    stride = Int(get(cfg["run"], "sample_stride", 1))
    statistics = log_density_contrast(fields[density_name]; stride)

    mach = NaN
    beta = NaN
    if isempty(filter(name -> !haskey(fields, name), physics_table_fields(cfg)))
        summary = physics_summary(fields, settings; stride)
        mach = summary.distributions[1].volume.mean
        beta = summary.distributions[3].volume.median
    else
        @info "Density PDF: no b parameter, the cube lacks velocity or magnetic fields"
    end
    b_value = driving_parameter(statistics.sigma, mach)
    b_magnetised = magnetised_driving_parameter(statistics.sigma, mach, beta)

    bins = Int(get(settings, "density_pdf_bins", 80))
    centers, counts, width = histogram_counts(statistics.contrast, bins)
    counts ./= sum(counts) * width

    figure = publication_figure(size=(960, 660))
    axis = latex_axis(figure[1, 1], xlabel=latexstring("s=\\ln(n/\\langle n\\rangle)"),
        ylabel=latexstring("p(s)"))
    barplot!(axis, centers, counts; width=0.94width, color=(PLOT_BLUE, 0.35),
        strokecolor=PLOT_BLUE, strokewidth=0.7,
        label=latex_legend_label("measured"))
    model = range(minimum(centers), maximum(centers); length=400)
    lines!(axis, model, lognormal_density.(model, statistics.sigma);
        color=PLOT_ORANGE, linewidth=2.8,
        label=latexstring("\\mathrm{lognormal},\\ \\sigma_s=",
            @sprintf("%.3f", statistics.sigma)))
    lines!(axis, model,
        gaussian_density.(model, statistics.mean_contrast, statistics.sigma);
        color=PLOT_TEAL, linewidth=2.2, linestyle=:dash,
        label=latexstring("\\mathrm{Gaussian\\ fit},\\ \\langle s\\rangle=",
            @sprintf("%.3f", statistics.mean_contrast)))
    if isfinite(b_value)
        vlines!(axis, [statistics.mean_contrast]; color=PLOT_MUTED,
            linestyle=:dot, linewidth=1.6,
            label=latexstring("b=", @sprintf("%.3f", b_value),
                ",\\ b_{B}=", isfinite(b_magnetised) ?
                    @sprintf("%.3f", b_magnetised) : "\\mathrm{n/a}"))
    end
    publication_legend!(axis; position=:lt)
    save_figure!(files, figure, output_dir, "density_pdf", formats; overwrite)
    save_data(cfg) && write_density_pdf!(files, statistics, mach, beta, b_value,
        b_magnetised, output_dir; overwrite)
    return files
end

"""
Volume and mass filling factors of each thermal phase, plus the mean density
and temperature inside it.
"""
function thermal_phase_budget(density, temperature, phases; stride=1)
    count_in = zeros(Int, length(phases))
    mass_in = zeros(Float64, length(phases))
    temperature_sum = zeros(Float64, length(phases))
    total_cells = 0
    total_mass = 0.0
    @inbounds for index in 1:stride:length(density)
        n = Float64(density[index])
        T = Float64(temperature[index])
        isfinite(n) && isfinite(T) && n > 0 || continue
        total_cells += 1
        total_mass += n
        phase_index = findfirst(phase -> phase.minimum <= T < phase.maximum, phases)
        isnothing(phase_index) && continue
        count_in[phase_index] += 1
        mass_in[phase_index] += n
        temperature_sum[phase_index] += T
    end
    total_cells > 0 || error("The thermal-phase budget has no valid cells")
    return [(name=phases[index].name, cells=count_in[index],
        volume_fraction=count_in[index] / total_cells,
        mass_fraction=total_mass > 0 ? mass_in[index] / total_mass : NaN,
        mean_density=count_in[index] > 0 ? mass_in[index] / count_in[index] : NaN,
        mean_temperature=count_in[index] > 0 ?
            temperature_sum[index] / count_in[index] : NaN)
        for index in eachindex(phases)]
end

function plot_phase_budget!(files, fields, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "diagnostics", Dict())
    density_name = string(get(settings, "density", "density"))
    temperature_name = string(get(settings, "temperature", "temperature"))
    if !haskey(fields, density_name) || !haskey(fields, temperature_name)
        @info "Skipping the phase budget; the cube lacks density or temperature"
        return files
    end
    phases = configured_thermal_phases(settings)
    stride = Int(get(cfg["run"], "sample_stride", 1))
    budget = thermal_phase_budget(fields[density_name], fields[temperature_name],
        phases; stride)

    positions = Float64.(eachindex(budget))
    figure = publication_figure(size=(960, 620))
    # Phase names as tick labels: latex_axis always installs a tickformat, so it
    # has to be overridden here rather than passing (values, labels) ticks.
    axis = latex_axis(figure[1, 1], xlabel=latex_text("thermal phase"),
        ylabel=latex_text("filling factor"), xticks=positions,
        xtickformat=ticks -> [begin
            index = round(Int, tick)
            1 <= index <= length(budget) ? latex_text(budget[index].name) :
                latexstring("")
        end for tick in ticks])
    barplot!(axis, positions .- 0.18, getfield.(budget, :volume_fraction);
        width=0.34, color=(PLOT_BLUE, 0.85), strokecolor=PLOT_BLUE,
        strokewidth=0.8, label=latex_legend_label("volume"))
    barplot!(axis, positions .+ 0.18, getfield.(budget, :mass_fraction);
        width=0.34, color=(PLOT_ORANGE, 0.85), strokecolor=PLOT_ORANGE,
        strokewidth=0.8, label=latex_legend_label("mass"))
    for (index, phase) in enumerate(budget)
        for (offset, value) in ((-0.18, phase.volume_fraction),
                (0.18, phase.mass_fraction))
            isfinite(value) || continue
            text!(axis, positions[index] + offset, value + 0.015;
                text=latexstring(@sprintf("%.3f", value)), align=(:center, :bottom),
                color=PLOT_INK, fontsize=14)
        end
    end
    ylims!(axis, 0, 1.12)
    publication_legend!(axis)
    save_figure!(files, figure, output_dir, "phase_budget", formats; overwrite)
    if save_data(cfg)
        path = joinpath(output_dir, "phase_budget.csv")
        write_text_output(path; overwrite) do io
            println(io, "phase,cells,volume_fraction,mass_fraction,mean_density,mean_temperature")
            for phase in budget
                @printf(io, "%s,%d,%.8e,%.8e,%.8e,%.8e\n", phase.name, phase.cells,
                    phase.volume_fraction, phase.mass_fraction, phase.mean_density,
                    phase.mean_temperature)
            end
        end
        push!(files, path)
    end
    return files
end

"""
Maximum-likelihood power-law tail fit following Clauset, Shalizi & Newman.

For each candidate lower bound the exponent is `alpha = 1 + n / sum ln(x/xmin)`;
the bound minimising the Kolmogorov-Smirnov distance to the fitted distribution
is retained, which avoids choosing the break by eye.
"""
function powerlaw_tail_fit(values; candidates=60, minimum_tail=50,
        minimum_fraction=0.02)
    sorted = sort!(Float64[v for v in values if isfinite(v) && v > 0])
    length(sorted) >= minimum_tail ||
        return (alpha=NaN, xmin=NaN, ks=NaN, tail=0)
    logs = log.(sorted)
    total = length(sorted)
    # Also require the tail to hold a fixed share of the sample, so that a small
    # map cannot have its exponent set by a handful of extreme pixels.
    minimum_tail = max(minimum_tail, round(Int, minimum_fraction * total))
    # Suffix sums of log(x) make each candidate an O(1) exponent evaluation.
    suffix = similar(logs)
    running = 0.0
    for index in total:-1:1
        running += logs[index]
        suffix[index] = running
    end
    lowest = max(1, searchsortedfirst(sorted, quantile(sorted, 0.30)))
    highest = max(lowest, total - minimum_tail + 1)
    step = max(1, (highest - lowest) ÷ max(candidates, 1))
    best = (alpha=NaN, xmin=NaN, ks=Inf, tail=0)
    for start in lowest:step:highest
        xmin = sorted[start]
        xmin > 0 || continue
        tail = total - start + 1
        tail >= minimum_tail || continue
        denominator = suffix[start] - tail * log(xmin)
        denominator > 0 || continue
        alpha = 1 + tail / denominator
        distance = 0.0
        for index in start:total
            empirical = (index - start + 1) / tail
            model = 1 - (sorted[index] / xmin)^(1 - alpha)
            distance = max(distance, abs(empirical - model))
        end
        distance < best.ks && (best = (alpha=alpha, xmin=xmin, ks=distance, tail=tail))
    end
    return isfinite(best.ks) ? best : (alpha=NaN, xmin=NaN, ks=NaN, tail=0)
end

function plot_column_density_pdf!(files, fields, metadata, cfg, output_dir, formats,
        overwrite)
    settings = get(cfg, "diagnostics", Dict())
    density_name = string(get(settings, "density", "density"))
    if !haskey(fields, density_name)
        @info "Skipping the column-density PDF; the cube lacks a density field"
        return files
    end
    axes_names = string.(get(settings, "column_density_axes", ["x", "y", "z"]))
    bins = Int(get(settings, "column_density_bins", 60))
    cube = fields[density_name]
    unit = column_density_unit(metadata[density_name].unit)

    figure = publication_figure(size=(980, 660))
    axis = latex_axis(figure[1, 1], xscale=log10, yscale=log10,
        xlabel=latex_text("column density [$unit]"), ylabel=latexstring("p(N)"))
    rows = NamedTuple[]
    for (index, name) in enumerate(axes_names)
        column = vec(column_density_map(cube, name, cfg))
        positive = Float64[value for value in column if isfinite(value) && value > 0]
        isempty(positive) && continue
        edges = 10 .^ range(log10(minimum(positive)), log10(maximum(positive));
            length=bins + 1)
        centers, counts, _ = histogram_counts(positive, edges)
        widths = diff(edges)
        density = counts ./ (sum(counts) .* widths)
        fit = powerlaw_tail_fit(positive)
        keep = findall(>(0), density)
        color = series_color(index)
        lines!(axis, centers[keep], density[keep]; color, linewidth=2.6,
            label=latex_legend_label("LOS $name"))
        if isfinite(fit.alpha)
            tail = findall(value -> value >= fit.xmin, centers)
            if !isempty(tail)
                anchor = first(tail)
                reference = density[anchor] > 0 ? density[anchor] :
                    maximum(density[keep])
                model = reference .* (centers[tail] ./ centers[anchor]) .^ (-fit.alpha)
                lines!(axis, centers[tail], model; color, linestyle=:dashdot,
                    linewidth=2.0,
                    label=latexstring("\\alpha_{", name, "}=",
                        @sprintf("%.2f", fit.alpha)))
            end
        end
        push!(rows, (axis=name, alpha=fit.alpha, xmin=fit.xmin, ks=fit.ks,
            tail=fit.tail, samples=length(positive)))
    end
    isempty(rows) && (@info "Column-density PDF: no positive column densities";
        return files)
    publication_legend!(axis; position=:lb)
    save_figure!(files, figure, output_dir, "column_density_pdf", formats; overwrite)
    if save_data(cfg)
        path = joinpath(output_dir, "column_density_pdf.csv")
        write_text_output(path; overwrite) do io
            println(io, "axis,samples,tail_samples,power_law_alpha,x_min,ks_distance")
            for row in rows
                @printf(io, "%s,%d,%d,%.8e,%.8e,%.8e\n", row.axis, row.samples,
                    row.tail, row.alpha, row.xmin, row.ks)
            end
        end
        push!(files, path)
    end
    return files
end

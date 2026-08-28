function plot_vector_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    for spec in get(cfg, "vector_spectra", Any[])
        name = string(spec["name"]); components = string.(spec["components"])
        transform = lowercase(string(get(spec, "transform", "none")))
        length(components) == 3 || error("Vector spectrum '$name' requires three components")
        all(haskey(fields, component) for component in components) ||
            error("Vector spectrum '$name' references missing fields")
        cells = prod(analysis_field_size(fields, first(components)))
        enforce_working_set(cfg, 24 * cells; context="vector FFT '$name'")
        weight_name = get(spec, "weight_field", nothing)
        !isnothing(weight_name) && !haskey(fields, string(weight_name)) &&
            error("Vector spectrum '$name' references missing weight field '$weight_name'")
        weight = isnothing(weight_name) ? nothing : fields[string(weight_name)]
        result = vector_spectrum_details((fields[c] for c in components)...;
            bins=Int(get(spec, "bins", 80)), box_size, axis_order,
            remove_mean=Bool(get(spec, "remove_mean", true)), weight,
            weight_power=Float64(get(spec, "weight_power", 0.5)), transform)
        if save_data(cfg)
            path = joinpath(output_dir, "vector_spectrum_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "k_physical,bin_width,total_shell,solenoidal_shell,compressive_shell,total_energy,solenoidal_energy,compressive_energy,compressive_fraction,modes")
                for i in eachindex(result.k)
                    fraction = result.total[i] > 0 ? result.compressive[i] / result.total[i] : NaN
                    @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d\n",
                        result.k[i], result.width[i], result.total[i],
                        result.solenoidal[i], result.compressive[i],
                        result.total_energy[i], result.solenoidal_energy[i],
                        result.compressive_energy[i], fraction, result.modes[i])
                end
            end
            push!(files, path)
        end
        quantity = string(get(spec, "quantity",
            get(get(cfg, "spectra", Dict()), "quantity", "energy")))
        quantity in ("energy", "shell") ||
            error("A vector-spectrum quantity must be energy or shell; got '$quantity'")
        total, solenoidal, compressive = quantity == "energy" ?
            (result.total_energy, result.solenoidal_energy, result.compressive_energy) :
            (result.total, result.solenoidal, result.compressive)
        box_unit = string(get(grid, "box_unit", "L"))
        spectrum_label = transform in ("curl", "vorticity") ?
            latexstring("\\log_{10}E_{\\omega}(k)") : latexstring("\\log_{10}E(k)")
        fig = publication_figure(); ax = latex_axis(fig[1, 1],
            xlabel=latexstring("\\log_{10}(k\\ [\\mathrm{", box_unit, "}^{-1}])"),
            ylabel=spectrum_label, xlogcoordinates=true, ylogcoordinates=true,
            )
        dissipation_cells = Float64(get(spec, "dissipation_cells",
            get(get(cfg, "spectra", Dict()), "dissipation_cells", 4.0)))
        injection_modes = Float64(get(spec, "injection_modes",
            get(get(cfg, "spectra", Dict()), "injection_modes", 4.0)))
        shape = analysis_field_size(fields, first(components))
        add_spectral_ranges!(ax, shape, box_size, axis_order;
            injection_modes, dissipation_cells)
        logk, logtotal = log_spectrum_coordinates(result.k, total)
        valid = findall(index -> isfinite(result.k[index]) && result.k[index] > 0 &&
            isfinite(total[index]) && total[index] > 0, eachindex(result.k, total))
        errorbars!(ax, logk, logtotal, log10_spectrum_uncertainty(result.modes[valid]);
            color=(PLOT_INK, 0.45), whiskerwidth=6, linewidth=1.1)
        lines!(ax, logk, logtotal; label=latex_legend_label("total"), linewidth=2.8,
            color=PLOT_INK)
        if transform == "none"
            logk, logsolenoidal = log_spectrum_coordinates(result.k, solenoidal)
            lines!(ax, logk, logsolenoidal; label=latex_legend_label("solenoidal"),
                linewidth=2.4, color=PLOT_BLUE)
            logk, logcompressive = log_spectrum_coordinates(result.k, compressive)
            lines!(ax, logk, logcompressive; label=latex_legend_label("compressive"),
                linewidth=2.4, color=PLOT_ORANGE, linestyle=:dash)
        end
        slopes = get(spec, "reference_slopes", Float64[])
        reference_range = inertial_fit_range(get(spec, "reference_range", nothing),
            shape, box_size, axis_order; injection_modes, dissipation_cells)
        add_reference_slopes!(ax, result.k, total, slopes; k_range=reference_range)
        publication_legend!(ax)
        save_figure!(files, fig, output_dir, "vector_spectrum_$(sanitize(name))", formats; overwrite)
    end
end

function plot_cross_spectra!(files, fields, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    for spec in get(cfg, "cross_spectra", Any[])
        name = string(spec["name"]); first = string(spec["first"]); second = string(spec["second"])
        haskey(fields, first) && haskey(fields, second) ||
            error("Cross spectrum '$name' references missing fields")
        enforce_working_set(cfg, 16 * prod(analysis_field_size(fields, first));
            context="cross FFT '$name'")
        result = cross_spectrum_details(fields[first], fields[second];
            bins=Int(get(spec, "bins", 80)), box_size, axis_order)
        if save_data(cfg)
            path = joinpath(output_dir, "cross_spectrum_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "k_physical,cross_real,cross_imag,cross_amplitude,coherence,modes")
                for i in eachindex(result.k)
                    @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%d\n", result.k[i],
                        real(result.cross[i]), imag(result.cross[i]), abs(result.cross[i]),
                        result.coherence[i], result.modes[i])
                end
            end
            push!(files, path)
        end
        fig = publication_figure(); ax = latex_axis(fig[1, 1], xscale=log10,
            xlabel=latexstring("k\\ [\\mathrm{physical}]"),
            ylabel=latexstring("\\mathcal{C}(k)"))
        dissipation_cells = Float64(get(spec, "dissipation_cells",
            get(get(cfg, "spectra", Dict()), "dissipation_cells", 4.0)))
        injection_modes = Float64(get(spec, "injection_modes",
            get(get(cfg, "spectra", Dict()), "injection_modes", 4.0)))
        add_spectral_ranges!(ax, analysis_field_size(fields, first), box_size,
            axis_order; injection_modes, dissipation_cells,
            logarithmic_coordinates=false)
        lines!(ax, result.k, result.coherence; linewidth=2.8, color=PLOT_BLUE)
        ylims!(ax, 0, 1.05)
        publication_legend!(ax)
        save_figure!(files, fig, output_dir, "cross_spectrum_$(sanitize(name))", formats; overwrite)
    end
end

function plot_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "spectra", Dict())
    names = string.(get(settings, "fields", collect(keys(fields))))
    bins = Int(get(settings, "bins", 80))
    remove_mean = Bool(get(settings, "remove_mean", true))
    compensation = Float64(get(settings, "compensation", 0.0))
    quantity = string(get(settings, "quantity", "energy"))
    quantity in ("energy", "average", "shell") ||
        error("spectra.quantity must be energy, average, or shell; got '$quantity'")
    periodic = Bool(get(get(cfg, "grid", Dict()), "periodic", true))
    window = string(get(settings, "window", periodic ? "none" : "hann"))
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    workspace = nothing
    for name in names
        haskey(fields, name) || error("Spectrum references missing field '$name'")
        cube = fields[name]
        shape = size(cube)
        enforce_memory_budget(fields, cfg; extra_bytes=12 * length(cube),
            context="FFT workspace for '$name'")
        isnothing(workspace) || size(workspace.input) == shape || (workspace = nothing)
        result = spectrum_details(cube; bins, remove_mean, box_size,
            axis_order, window, workspace)
        workspace = result.workspace
        power = spectrum_quantity(result, quantity)
        compensated = power .* result.k .^ compensation
        dissipation_cells = Float64(get(settings, "dissipation_cells", 4.0))
        injection_modes = Float64(get(settings, "injection_modes", 4.0))
        physical_fit_range = inertial_fit_range(get(settings, "fit_range", nothing),
            shape, box_size, axis_order; injection_modes, dissipation_cells)
        fit_range = selected_fit_range(result.k, power, physical_fit_range, settings)
        slope, slope_std, fit_modes = spectral_fit(result.k, power, fit_range;
            weights=sqrt.(result.modes))
        if save_data(cfg)
            path = joinpath(output_dir, "spectrum_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "k_physical,bin_width,power_mean,power_shell,power_energy,modes,compensated")
                for index in eachindex(result.k)
                    @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%d,%.8e\n", result.k[index],
                        result.width[index], result.average[index], result.shell[index],
                        result.energy[index], result.modes[index], compensated[index])
                end
            end
            push!(files, path)
            fit_path = joinpath(output_dir, "spectrum_$(sanitize(name))_fit.toml")
            write_text_output(fit_path; overwrite) do io
                TOML.print(io, Dict("slope" => slope, "slope_std" => slope_std,
                    "points" => fit_modes, "k_min" => fit_range[1], "k_max" => fit_range[2],
                    "quantity" => quantity, "window" => window); sorted=true)
            end
            push!(files, fit_path)
        end
        fig = publication_figure()
        symbol = quantity == "energy" ? "E" : quantity == "average" ? "P" : "S"
        ylabel = compensation == 0 ? latexstring("\\log_{10}", symbol, "(k)") :
            latexstring("\\log_{10}[k^{", compensation, "}", symbol, "(k)]")
        box_unit = string(get(get(cfg, "grid", Dict()), "box_unit", "L"))
        ax = latex_axis(fig[1, 1],
            xlabel=latexstring("\\log_{10}(k\\ [\\mathrm{", box_unit, "}^{-1}])"),
            ylabel=ylabel, xlogcoordinates=true, ylogcoordinates=true)
        add_spectral_ranges!(ax, shape, box_size, axis_order;
            injection_modes, dissipation_cells)
        logk, logpower = log_spectrum_coordinates(result.k, compensated)
        valid = findall(index -> isfinite(result.k[index]) && result.k[index] > 0 &&
            isfinite(compensated[index]) && compensated[index] > 0,
            eachindex(result.k, compensated))
        errorbars!(ax, logk, logpower,
            log10_spectrum_uncertainty(result.modes[valid]);
            color=(PLOT_BLUE, 0.55), whiskerwidth=6, linewidth=1.1)
        lines!(ax, logk, logpower; color=PLOT_BLUE, linewidth=2.7)
        scatter!(ax, logk, logpower; color=:white, strokecolor=PLOT_BLUE,
            strokewidth=1.2, markersize=6)
        add_spectral_fit!(ax, result.k, compensated, slope, slope_std, fit_range;
            compensation)
        if name in string.(get(settings, "velocity_fields", ["Vmag"]))
            slopes = get(settings, "velocity_reference_slopes", [-5 / 3, -2.0])
            reference_range = inertial_fit_range(
                get(settings, "reference_range", fit_range), shape,
                box_size, axis_order; injection_modes, dissipation_cells)
            add_reference_slopes!(ax, result.k, compensated, slopes;
                k_range=reference_range, compensation)
        end
        publication_legend!(ax)
        save_figure!(files, fig, output_dir, "spectrum_$(sanitize(name))", formats; overwrite)
    end
end

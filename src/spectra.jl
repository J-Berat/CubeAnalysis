mutable struct SpectrumWorkspace
    input::Array{Float32,3}
    output::Array{ComplexF32,3}
    plan
end

function SpectrumWorkspace(shape::NTuple{3,Int})
    input = zeros(Float32, shape)
    plan = plan_rfft(input; flags=FFTW.ESTIMATE)
    output = plan * input
    return SpectrumWorkspace(input, output, plan)
end

function prepare_spectrum_input!(workspace::SpectrumWorkspace, cube; remove_mean=true,
        window="none")
    size(workspace.input) == size(cube) || error("Spectrum workspace has the wrong size")
    finite_sum = 0.0; finite_count = 0
    @inbounds for value in cube
        if isfinite(value)
            finite_sum += value; finite_count += 1
        end
    end
    finite_count > 0 || error("Cannot compute a spectrum without finite values")
    replacement = Float32(finite_sum / finite_count)
    @inbounds for index in eachindex(cube)
        value = Float32(cube[index])
        workspace.input[index] = isfinite(value) ? value : replacement
    end
    remove_mean && (workspace.input .-= mean(workspace.input))
    if window == "hann"
        windows = ntuple(d -> Float32.(0.5 .- 0.5 .* cos.(2π .* (0:size(cube, d)-1) ./
            max(size(cube, d) - 1, 1))), 3)
        @inbounds for k in axes(cube, 3), j in axes(cube, 2), i in axes(cube, 1)
            workspace.input[i, j, k] *= windows[1][i] * windows[2][j] * windows[3][k]
        end
    elseif window != "none"
        error("Spectrum window must be 'none' or 'hann'")
    end
    mul!(workspace.output, workspace.plan, workspace.input)
    workspace.output ./= sqrt(Float32(length(cube)))
    return workspace
end

function spectrum_details(cube; bins::Int=80, remove_mean::Bool=true,
        box_size=(2π, 2π, 2π), axis_order=("x", "y", "z"), window="none",
        workspace=nothing)
    bins > 1 || error("Spectrum bins must be > 1")
    lengths = box_lengths(box_size, axis_order)
    ws = isnothing(workspace) ? SpectrumWorkspace(size(cube)) : workspace
    prepare_spectrum_input!(ws, cube; remove_mean, window)
    modes = (
        FFTW.rfftfreq(size(cube, 1)) .* size(cube, 1),
        FFTW.fftfreq(size(cube, 2)) .* size(cube, 2),
        FFTW.fftfreq(size(cube, 3)) .* size(cube, 3),
    )
    wave = ntuple(d -> 2π .* modes[d] ./ lengths[d], 3)
    kmin = minimum(2π ./ collect(lengths))
    kmax = minimum(π .* collect(size(cube)) ./ collect(lengths))
    kmax > kmin || error("Cube is too small for an isotropic spectrum")
    edges = 10 .^ range(log10(kmin), log10(kmax); length=bins + 1)
    sums = zeros(Float64, bins); counts = zeros(Int, bins); weighted_modes = zeros(Int, bins)
    n1 = size(cube, 1)
    @inbounds for k in axes(ws.output, 3), j in axes(ws.output, 2), i in axes(ws.output, 1)
        radius = sqrt(wave[1][i]^2 + wave[2][j]^2 + wave[3][k]^2)
        kmin <= radius <= kmax || continue
        bin = clamp(searchsortedlast(edges, radius), 1, bins)
        hermitian_weight = (i == 1 || (iseven(n1) && i == size(ws.output, 1))) ? 1 : 2
        sums[bin] += hermitian_weight * abs2(ws.output[i, j, k])
        counts[bin] += 1
        weighted_modes[bin] += hermitian_weight
    end
    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    keep = weighted_modes .> 0
    shell = sums[keep]
    average = shell ./ weighted_modes[keep]
    return (k=centers[keep], average=average, shell=shell,
        modes=weighted_modes[keep], stored_modes=counts[keep], workspace=ws)
end

function isotropic_spectrum(cube; bins::Int=80, remove_mean::Bool=true,
        box_size=(2π, 2π, 2π), axis_order=("x", "y", "z"), window="none")
    result = spectrum_details(cube; bins, remove_mean, box_size, axis_order, window)
    return result.k, result.average, result.modes
end

function spectral_fit(k, power, fit_range)
    selected = findall(i -> fit_range[1] <= k[i] <= fit_range[2] && power[i] > 0, eachindex(k))
    length(selected) >= 3 || return (NaN, NaN, length(selected))
    x = log10.(k[selected]); y = log10.(power[selected])
    xmean = mean(x); ymean = mean(y)
    denominator = sum(abs2, x .- xmean)
    slope = sum((x .- xmean) .* (y .- ymean)) / denominator
    intercept = ymean - slope * xmean
    residual = y .- (intercept .+ slope .* x)
    stderr = length(x) > 2 ? sqrt(sum(abs2, residual) / (length(x) - 2) / denominator) : NaN
    return slope, stderr, length(selected)
end

function vector_spectrum_details(vx, vy, vz; bins=80, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"), remove_mean=true, weight=nothing,
        weight_power=0.5, transform="none")
    transform = lowercase(string(transform))
    transform in ("none", "curl", "vorticity") ||
        error("Vector-spectrum transform must be none, curl, or vorticity")
    shape = size(vx)
    size(vy) == shape && size(vz) == shape || error("Vector components must have equal sizes")
    fourier = Array{ComplexF32,3}[]
    for component in (vx, vy, vz)
        A = Float32.(component)
        if !isnothing(weight)
            @inbounds for i in eachindex(A)
                w = Float32(weight[i])
                A[i] = isfinite(w) && w >= 0 ? A[i] * w^weight_power : NaN32
            end
        end
        finite = filter(isfinite, A)
        isempty(finite) && error("Vector spectrum requires finite values")
        replacement = mean(finite)
        @inbounds for i in eachindex(A)
            isfinite(A[i]) || (A[i] = replacement)
        end
        remove_mean && (A .-= mean(A))
        push!(fourier, rfft(A) ./ sqrt(Float32(length(A))))
    end
    lengths = box_lengths(box_size, axis_order)
    mode_axes = (FFTW.rfftfreq(shape[1]) .* shape[1], FFTW.fftfreq(shape[2]) .* shape[2],
        FFTW.fftfreq(shape[3]) .* shape[3])
    wave = ntuple(d -> 2π .* mode_axes[d] ./ lengths[d], 3)
    kmin = minimum(2π ./ collect(lengths)); kmax = minimum(π .* collect(shape) ./ collect(lengths))
    edges = 10 .^ range(log10(kmin), log10(kmax); length=bins + 1)
    total = zeros(Float64, bins); solenoidal = zeros(Float64, bins)
    compressive = zeros(Float64, bins); modes = zeros(Int, bins)
    @inbounds for k in axes(fourier[1], 3), j in axes(fourier[1], 2), i in axes(fourier[1], 1)
        kx, ky, kz = wave[1][i], wave[2][j], wave[3][k]
        k2 = kx^2 + ky^2 + kz^2
        kmin^2 <= k2 <= kmax^2 || continue
        bin = clamp(searchsortedlast(edges, sqrt(k2)), 1, bins)
        weight = (i == 1 || (iseven(shape[1]) && i == size(fourier[1], 1))) ? 1 : 2
        fx, fy, fz = fourier[1][i, j, k], fourier[2][i, j, k], fourier[3][i, j, k]
        if transform in ("curl", "vorticity")
            fx, fy, fz = im * (ky * fz - kz * fy), im * (kz * fx - kx * fz),
                im * (kx * fy - ky * fx)
        end
        energy = abs2(fx) + abs2(fy) + abs2(fz)
        longitudinal = abs2(kx * fx + ky * fy + kz * fz) / k2
        total[bin] += weight * energy
        compressive[bin] += weight * longitudinal
        solenoidal[bin] += weight * max(energy - longitudinal, 0)
        modes[bin] += weight
    end
    centers = sqrt.(edges[1:end-1] .* edges[2:end]); keep = modes .> 0
    return (k=centers[keep], total=total[keep], solenoidal=solenoidal[keep],
        compressive=compressive[keep], modes=modes[keep])
end

function log_spectrum_coordinates(k, power)
    valid = findall(index -> isfinite(k[index]) && k[index] > 0 &&
        isfinite(power[index]) && power[index] > 0, eachindex(k, power))
    return log10.(k[valid]), log10.(power[valid])
end

function nyquist_wavenumber(shape, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"))
    lengths = box_lengths(box_size, axis_order)
    return minimum(π .* collect(shape) ./ collect(lengths))
end

function dissipation_wavenumber(shape, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"); cells=4.0)
    cells = Float64(cells)
    cells >= 2 || error("spectra.dissipation_cells must be at least 2")
    return 2 * nyquist_wavenumber(shape, box_size, axis_order) / cells
end

function add_dissipation_region!(axis, shape, box_size, axis_order; cells=4.0,
        logarithmic_coordinates=true)
    nyquist = nyquist_wavenumber(shape, box_size, axis_order)
    dissipation = dissipation_wavenumber(shape, box_size, axis_order; cells)
    left, right = logarithmic_coordinates ? log10.((dissipation, nyquist)) :
        (dissipation, nyquist)
    vspan!(axis, left, right; color=(:gray55, 0.25),
        label=latexstring("\\mathrm{dissipation/noise}"))
    vlines!(axis, [left]; color=:gray25, linestyle=:dashdot, linewidth=2.0,
        label=latexstring("k_{\\mathrm{diss}}"))
    return dissipation
end

function reference_power_law(k, power, slope; k_range=nothing, offset=0.0)
    valid = findall(index -> isfinite(k[index]) && k[index] > 0 &&
        isfinite(power[index]) && power[index] > 0, eachindex(k, power))
    if !isnothing(k_range)
        limits = Float64.(k_range)
        length(limits) == 2 || error("A reference-slope range needs two values")
        valid = filter(index -> limits[1] <= k[index] <= limits[2], valid)
    end
    length(valid) >= 2 || return (Float64[], Float64[])
    x = log10.(Float64.(k[valid]))
    observed = log10.(Float64.(power[valid]))
    anchor = cld(length(x), 2)
    y = observed[anchor] + Float64(offset) .+ Float64(slope) .* (x .- x[anchor])
    return x, y
end

function reference_slope_label(slope)
    isapprox(slope, -5 / 3; atol=1e-4) &&
        return latexstring("\\mathrm{Kolmogorov}\\ k^{-5/3}")
    isapprox(slope, -2; atol=1e-4) && return latexstring("\\mathrm{Burgers}\\ k^{-2}")
    return latexstring("k^{", @sprintf("%.3g", slope), "}")
end

function add_reference_slopes!(axis, k, power, slopes; k_range=nothing,
        compensation=0.0)
    colors = (:crimson, :royalblue, :darkgreen, :purple)
    count = length(slopes)
    for (index, physical_slope) in enumerate(Float64.(slopes))
        displayed_slope = physical_slope + compensation
        offset = 0.20 * (index - (count + 1) / 2)
        x, y = reference_power_law(k, power, displayed_slope; k_range, offset)
        isempty(x) && continue
        lines!(axis, x, y; color=colors[mod1(index, length(colors))],
            linestyle=:dash, linewidth=2.0, label=reference_slope_label(physical_slope))
    end
    return axis
end

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
                println(io, "k_physical,total_shell,solenoidal_shell,compressive_shell,compressive_fraction,modes")
                for i in eachindex(result.k)
                    fraction = result.total[i] > 0 ? result.compressive[i] / result.total[i] : NaN
                    @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%d\n", result.k[i], result.total[i],
                        result.solenoidal[i], result.compressive[i], fraction, result.modes[i])
                end
            end
            push!(files, path)
        end
        box_unit = string(get(grid, "box_unit", "L"))
        spectrum_label = transform in ("curl", "vorticity") ?
            latexstring("\\log_{10}E_{\\omega}(k)") : latexstring("\\log_{10}E(k)")
        fig = Figure(size=(760, 520)); ax = latex_axis(fig[1, 1],
            xlabel=latexstring("\\log_{10}(k\\ [\\mathrm{", box_unit, "}^{-1}])"),
            ylabel=spectrum_label, xlogcoordinates=true, ylogcoordinates=true,
            title=replace(name, '_' => ' '))
        dissipation_cells = Float64(get(spec, "dissipation_cells",
            get(get(cfg, "spectra", Dict()), "dissipation_cells", 4.0)))
        add_dissipation_region!(ax, analysis_field_size(fields, first(components)),
            box_size, axis_order; cells=dissipation_cells)
        logk, logtotal = log_spectrum_coordinates(result.k, result.total)
        lines!(ax, logk, logtotal; label=latex_legend_label("total"), linewidth=2.5)
        if transform == "none"
            logk, logsolenoidal = log_spectrum_coordinates(result.k, result.solenoidal)
            lines!(ax, logk, logsolenoidal; label=latex_legend_label("solenoidal"), linewidth=2)
            logk, logcompressive = log_spectrum_coordinates(result.k, result.compressive)
            lines!(ax, logk, logcompressive; label=latex_legend_label("compressive"), linewidth=2)
        end
        slopes = get(spec, "reference_slopes", Float64[])
        add_reference_slopes!(ax, result.k, result.total, slopes;
            k_range=get(spec, "reference_range", nothing))
        axislegend(ax)
        save_figure!(files, fig, output_dir, "vector_spectrum_$(sanitize(name))", formats; overwrite)
    end
end

function cross_spectrum_details(first, second; bins=80, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"))
    size(first) == size(second) || error("Cross-spectrum fields must have equal sizes")
    transformed = Array{ComplexF32,3}[]
    for source in (first, second)
        A = Float32.(source)
        finite = filter(isfinite, A); isempty(finite) && error("Cross spectrum requires finite values")
        replacement = mean(finite)
        @inbounds for i in eachindex(A); isfinite(A[i]) || (A[i] = replacement); end
        A .-= mean(A)
        push!(transformed, rfft(A) ./ sqrt(Float32(length(A))))
    end
    shape = size(first); lengths = box_lengths(box_size, axis_order)
    modes_axes = (FFTW.rfftfreq(shape[1]) .* shape[1], FFTW.fftfreq(shape[2]) .* shape[2],
        FFTW.fftfreq(shape[3]) .* shape[3])
    wave = ntuple(d -> 2π .* modes_axes[d] ./ lengths[d], 3)
    kmin = minimum(2π ./ collect(lengths)); kmax = minimum(π .* collect(shape) ./ collect(lengths))
    edges = 10 .^ range(log10(kmin), log10(kmax); length=bins + 1)
    cross = zeros(ComplexF64, bins); first_power = zeros(Float64, bins)
    second_power = zeros(Float64, bins); modes = zeros(Int, bins)
    @inbounds for k in axes(transformed[1], 3), j in axes(transformed[1], 2), i in axes(transformed[1], 1)
        radius = sqrt(wave[1][i]^2 + wave[2][j]^2 + wave[3][k]^2)
        kmin <= radius <= kmax || continue
        bin = clamp(searchsortedlast(edges, radius), 1, bins)
        weight = (i == 1 || (iseven(shape[1]) && i == size(transformed[1], 1))) ? 1 : 2
        a, b = transformed[1][i, j, k], transformed[2][i, j, k]
        cross[bin] += weight * a * conj(b)
        first_power[bin] += weight * abs2(a); second_power[bin] += weight * abs2(b)
        modes[bin] += weight
    end
    centers = sqrt.(edges[1:end-1] .* edges[2:end]); keep = modes .> 0
    coherence = abs2.(cross[keep]) ./ max.(first_power[keep] .* second_power[keep], eps())
    return (k=centers[keep], cross=cross[keep], coherence, modes=modes[keep])
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
        fig = Figure(size=(760, 520)); ax = latex_axis(fig[1, 1], xscale=log10,
            xlabel=latexstring("k\\ [\\mathrm{physical}]"),
            ylabel=latexstring("\\mathcal{C}(k)"), title=replace(name, '_' => ' '))
        dissipation_cells = Float64(get(spec, "dissipation_cells",
            get(get(cfg, "spectra", Dict()), "dissipation_cells", 4.0)))
        add_dissipation_region!(ax, analysis_field_size(fields, first), box_size,
            axis_order; cells=dissipation_cells, logarithmic_coordinates=false)
        lines!(ax, result.k, result.coherence; linewidth=2.5); ylims!(ax, 0, 1.05)
        axislegend(ax)
        save_figure!(files, fig, output_dir, "cross_spectrum_$(sanitize(name))", formats; overwrite)
    end
end

function plot_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "spectra", Dict())
    names = string.(get(settings, "fields", collect(keys(fields))))
    bins = Int(get(settings, "bins", 80))
    remove_mean = Bool(get(settings, "remove_mean", true))
    compensation = Float64(get(settings, "compensation", 0.0))
    quantity = string(get(settings, "quantity", "average"))
    quantity in ("average", "shell") || error("spectra.quantity must be average or shell")
    periodic = Bool(get(get(cfg, "grid", Dict()), "periodic", true))
    window = string(get(settings, "window", periodic ? "none" : "hann"))
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    workspace = nothing
    for name in names
        haskey(fields, name) || error("Spectrum references missing field '$name'")
        cells = length(fields[name])
        enforce_memory_budget(fields, cfg; extra_bytes=12 * cells,
            context="FFT workspace for '$name'")
        size(fields[name]) == (isnothing(workspace) ? size(fields[name]) : size(workspace.input)) ||
            (workspace = nothing)
        result = spectrum_details(fields[name]; bins, remove_mean, box_size,
            axis_order, window, workspace)
        workspace = result.workspace
        power = quantity == "average" ? result.average : result.shell
        compensated = power .* result.k .^ compensation
        fit_range = Float64.(get(settings, "fit_range", [minimum(result.k), maximum(result.k)]))
        length(fit_range) == 2 || error("spectra.fit_range needs two values")
        slope, slope_std, fit_modes = spectral_fit(result.k, power, fit_range)
        if save_data(cfg)
            path = joinpath(output_dir, "spectrum_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "k_physical,power_mean,power_shell,modes,compensated")
                for index in eachindex(result.k)
                    @printf(io, "%.8e,%.8e,%.8e,%d,%.8e\n", result.k[index],
                        result.average[index], result.shell[index], result.modes[index], compensated[index])
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
        fig = Figure(size=(760, 520))
        ylabel = compensation == 0 ? latexstring("\\log_{10}P(k)") :
            latexstring("\\log_{10}[k^{", compensation, "}P(k)]")
        box_unit = string(get(get(cfg, "grid", Dict()), "box_unit", "L"))
        ax = latex_axis(fig[1, 1], title="3-D isotropic spectrum - $(metadata[name].label)",
            xlabel=latexstring("\\log_{10}(k\\ [\\mathrm{", box_unit, "}^{-1}])"),
            ylabel=ylabel, xlogcoordinates=true, ylogcoordinates=true)
        dissipation_cells = Float64(get(settings, "dissipation_cells", 4.0))
        add_dissipation_region!(ax, size(fields[name]), box_size, axis_order;
            cells=dissipation_cells)
        logk, logpower = log_spectrum_coordinates(result.k, compensated)
        lines!(ax, logk, logpower; color=:darkorange, linewidth=2.5)
        scatter!(ax, logk, logpower; color=:darkorange, markersize=6)
        if name in string.(get(settings, "velocity_fields", ["Vmag"]))
            slopes = get(settings, "velocity_reference_slopes", [-5 / 3, -2.0])
            add_reference_slopes!(ax, result.k, compensated, slopes;
                k_range=get(settings, "reference_range", fit_range), compensation)
        end
        axislegend(ax)
        save_figure!(files, fig, output_dir, "spectrum_$(sanitize(name))", formats; overwrite)
    end
end

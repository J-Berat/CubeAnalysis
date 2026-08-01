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
        axis_order=("x", "y", "z"), remove_mean=true, weight=nothing, weight_power=0.5)
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

function plot_vector_spectra!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    for spec in get(cfg, "vector_spectra", Any[])
        name = string(spec["name"]); components = string.(spec["components"])
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
            weight_power=Float64(get(spec, "weight_power", 0.5)))
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
        fig = Figure(size=(760, 520)); ax = latex_axis(fig[1, 1], xscale=log10, yscale=log10,
            xlabel="k [physical]", ylabel="shell energy", title=replace(name, '_' => ' '))
        lines!(ax, result.k, result.total; label=latex_legend_label("total"), linewidth=2.5)
        lines!(ax, result.k, result.solenoidal; label=latex_legend_label("solenoidal"), linewidth=2)
        lines!(ax, result.k, result.compressive; label=latex_legend_label("compressive"), linewidth=2)
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
        fig = Figure(size=(760, 520)); ax = latex_axis(fig[1, 1], xscale=log10,
            xlabel="k [physical]", ylabel="coherence", title=replace(name, '_' => ' '))
        lines!(ax, result.k, result.coherence; linewidth=2.5); ylims!(ax, 0, 1.05)
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
        fig = Figure(size=(760, 520))
        ylabel = compensation == 0 ? "P(k)" : "k^$(compensation) P(k)"
        box_unit = string(get(get(cfg, "grid", Dict()), "box_unit", "L"))
        ax = latex_axis(fig[1, 1], title="3-D isotropic spectrum - $(metadata[name].label)",
            xlabel="k [1 / $(box_unit)]",
            ylabel=ylabel, xscale=log10, yscale=log10)
        lines!(ax, result.k, compensated; color=:darkorange, linewidth=2.5)
        scatter!(ax, result.k, compensated; color=:darkorange, markersize=6)
        save_figure!(files, fig, output_dir, "spectrum_$(sanitize(name))", formats; overwrite)
    end
end

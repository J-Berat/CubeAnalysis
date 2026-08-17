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
    width = diff(edges)[keep]
    return (k=centers[keep], average=average, shell=shell, energy=shell ./ width,
        width=width, modes=weighted_modes[keep], stored_modes=counts[keep],
        workspace=ws)
end

"""
Pick the plotted quantity from a spectrum result.

- `"energy"` is the shell-integrated spectral density `E(k) = sum|F|^2 / dk`.
  Because the bins are logarithmic, `dk` grows like `k`, so only this
  normalisation can be compared with the Kolmogorov `k^-5/3` or Burgers `k^-2`
  references.
- `"shell"` is the raw sum over the bin, whose slope is that of `E(k)` plus one.
- `"average"` is the power per Fourier mode, whose slope is that of `E(k)`
  minus two (`-11/3` for Kolmogorov).
"""
function spectrum_quantity(result, quantity::AbstractString)
    quantity == "energy" && return result.energy
    quantity == "shell" && return result.shell
    quantity == "average" && return result.average
    error("spectra.quantity must be energy, shell, or average; got '$quantity'")
end

function isotropic_spectrum(cube; bins::Int=80, remove_mean::Bool=true,
        box_size=(2π, 2π, 2π), axis_order=("x", "y", "z"), window="none")
    result = spectrum_details(cube; bins, remove_mean, box_size, axis_order, window)
    return result.k, result.average, result.modes
end

"""
Least-squares power-law slope of `power` against `k` over `fit_range`.

`weights` gives the relative confidence of each bin. For a shell-averaged
spectrum the natural choice is `sqrt(modes)`: logarithmic bins hold a number of
Fourier modes growing like `k^3`, so an unweighted fit lets the handful of modes
at low `k` count as much as the 10^5 modes near the Nyquist frequency.
"""
function spectral_fit(k, power, fit_range; weights=nothing)
    selected = findall(i -> fit_range[1] <= k[i] <= fit_range[2] && power[i] > 0 &&
        isfinite(power[i]) && k[i] > 0, eachindex(k))
    length(selected) >= 3 || return (NaN, NaN, length(selected))
    x = log10.(Float64.(k[selected])); y = log10.(Float64.(power[selected]))
    w = isnothing(weights) ? ones(length(selected)) :
        Float64.(weights[selected])
    all(isfinite, w) && all(>=(0), w) && sum(w) > 0 ||
        error("Spectral-fit weights must be finite and non-negative")
    total = sum(w)
    xmean = sum(w .* x) / total; ymean = sum(w .* y) / total
    denominator = sum(w .* abs2.(x .- xmean))
    denominator > 0 || return (NaN, NaN, length(selected))
    slope = sum(w .* (x .- xmean) .* (y .- ymean)) / denominator
    intercept = ymean - slope * xmean
    residual = y .- (intercept .+ slope .* x)
    stderr = length(x) > 2 ?
        sqrt(sum(w .* abs2.(residual)) / (length(x) - 2) / denominator) : NaN
    return slope, stderr, length(selected)
end

"""
One-sigma uncertainty of a shell-averaged spectrum in log10 units.

Each bin averages `modes` independent Fourier coefficients, so the relative
error on the power is `1/sqrt(modes)`; in log10 that is a constant offset.
"""
log10_spectrum_uncertainty(modes) =
    [count > 0 ? 1 / (log(10) * sqrt(count)) : NaN for count in modes]

function add_spectral_fit!(axis, k, displayed_power, slope, slope_std, fit_range;
        compensation=0.0)
    isfinite(slope) || return axis
    x, y = reference_power_law(k, displayed_power, slope + compensation;
        k_range=fit_range)
    isempty(x) && return axis
    uncertainty = isfinite(slope_std) ? @sprintf("%.2f", slope_std) : "\\mathrm{nan}"
    label = latexstring("\\alpha=", @sprintf("%.2f", slope),
        "\\pm", uncertainty)
    lines!(axis, x, y; color=PLOT_INK, linestyle=:dashdot, linewidth=2.2,
        label)
    return axis
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
    plan = plan_rfft(Array{Float32,3}(undef, shape); flags=FFTW.ESTIMATE)
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
        push!(fourier, (plan * A) ./ sqrt(Float32(length(A))))
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
    width = diff(edges)[keep]
    return (k=centers[keep], total=total[keep], solenoidal=solenoidal[keep],
        compressive=compressive[keep], modes=modes[keep], width=width,
        total_energy=total[keep] ./ width,
        solenoidal_energy=solenoidal[keep] ./ width,
        compressive_energy=compressive[keep] ./ width)
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

function fundamental_wavenumber(box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"))
    lengths = box_lengths(box_size, axis_order)
    return minimum(2π ./ collect(lengths))
end

function injection_wavenumber(box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"); modes=4.0)
    modes = Float64(modes)
    modes >= 1 || error("spectra.injection_modes must be at least 1")
    return modes * fundamental_wavenumber(box_size, axis_order)
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
    vspan!(axis, left, right; color=(PLOT_MUTED, 0.16),
        label=latexstring("\\mathrm{dissipation/noise}"))
    vlines!(axis, [left]; color=PLOT_MUTED, linestyle=:dashdot, linewidth=1.8,
        label=latexstring("k_{\\mathrm{diss}}"))
    return dissipation
end


function add_injection_region!(axis, shape, box_size, axis_order; modes=4.0,
        logarithmic_coordinates=true)
    fundamental = fundamental_wavenumber(box_size, axis_order)
    injection = injection_wavenumber(box_size, axis_order; modes)
    upper = min(injection, nyquist_wavenumber(shape, box_size, axis_order))
    left, right = logarithmic_coordinates ? log10.((fundamental, upper)) :
        (fundamental, upper)
    right > left && vspan!(axis, left, right; color=(PLOT_GOLD, 0.18),
        label=latexstring("\\mathrm{energy\\ injection}"))
    position = logarithmic_coordinates ? log10(injection) : injection
    vlines!(axis, [position]; color=PLOT_ORANGE, linestyle=:dashdot, linewidth=1.8,
        label=latexstring("k_{\\mathrm{inj}}"))
    return injection
end

function add_spectral_ranges!(axis, shape, box_size, axis_order; injection_modes=4.0,
        dissipation_cells=4.0, logarithmic_coordinates=true)
    injection = add_injection_region!(axis, shape, box_size, axis_order;
        modes=injection_modes, logarithmic_coordinates)
    dissipation = add_dissipation_region!(axis, shape, box_size, axis_order;
        cells=dissipation_cells, logarithmic_coordinates)
    return (injection, dissipation)
end

function inertial_fit_range(requested, shape, box_size, axis_order;
        injection_modes=4.0, dissipation_cells=4.0)
    injection = injection_wavenumber(box_size, axis_order; modes=injection_modes)
    dissipation = dissipation_wavenumber(shape, box_size, axis_order;
        cells=dissipation_cells)
    limits = isnothing(requested) ? (-Inf, Inf) : Tuple(Float64.(requested))
    length(limits) == 2 || error("A spectral fit range needs two values")
    return [max(limits[1], injection), min(limits[2], dissipation)]
end

"""Select the longest well-described power-law window inside physical bounds."""
function detect_powerlaw_range(x, y, bounds; min_points=6, min_decades=0.35,
        min_r2=0.95)
    candidates = findall(i -> isfinite(x[i]) && x[i] > 0 && isfinite(y[i]) &&
        y[i] > 0 && bounds[1] <= x[i] <= bounds[2], eachindex(x, y))
    length(candidates) >= min_points || return Float64.(bounds)
    lx = log10.(Float64.(x[candidates])); ly = log10.(Float64.(y[candidates]))
    best = nothing
    for first in 1:length(candidates)-min_points+1, last in first+min_points-1:length(candidates)
        span = lx[last] - lx[first]; span >= min_decades || continue
        xx = @view lx[first:last]; yy = @view ly[first:last]
        xm, ym = mean(xx), mean(yy); denominator = sum(abs2, xx .- xm)
        denominator > 0 || continue
        slope = sum((xx .- xm) .* (yy .- ym)) / denominator
        residual = yy .- (ym - slope * xm .+ slope .* xx)
        total = sum(abs2, yy .- ym)
        r2 = total > 0 ? 1 - sum(abs2, residual) / total : 1.0
        r2 >= min_r2 || continue
        score = (span, r2, last - first + 1)
        (isnothing(best) || score > best.score) &&
            (best=(score=score, first=first, last=last))
    end
    isnothing(best) && return Float64.(bounds)
    return [Float64(x[candidates[best.first]]), Float64(x[candidates[best.last]])]
end

function selected_fit_range(x, y, physical_range, settings)
    Bool(get(settings, "auto_fit_range", true)) || return physical_range
    return detect_powerlaw_range(x, y, physical_range;
        min_points=Int(get(settings, "auto_fit_min_points", 6)),
        min_decades=Float64(get(settings, "auto_fit_min_decades", 0.35)),
        min_r2=Float64(get(settings, "auto_fit_min_r2", 0.95)))
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
    count = length(slopes)
    for (index, physical_slope) in enumerate(Float64.(slopes))
        displayed_slope = physical_slope + compensation
        offset = 0.20 * (index - (count + 1) / 2)
        x, y = reference_power_law(k, power, displayed_slope; k_range, offset)
        isempty(x) && continue
        lines!(axis, x, y; color=series_color(index + 1),
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

function cross_spectrum_details(first, second; bins=80, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"))
    size(first) == size(second) || error("Cross-spectrum fields must have equal sizes")
    transformed = Array{ComplexF32,3}[]
    plan = plan_rfft(Array{Float32,3}(undef, size(first)); flags=FFTW.ESTIMATE)
    for source in (first, second)
        A = Float32.(source)
        finite = filter(isfinite, A); isempty(finite) && error("Cross spectrum requires finite values")
        replacement = mean(finite)
        @inbounds for i in eachindex(A); isfinite(A[i]) || (A[i] = replacement); end
        A .-= mean(A)
        push!(transformed, (plan * A) ./ sqrt(Float32(length(A))))
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

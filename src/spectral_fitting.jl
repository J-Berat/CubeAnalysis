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


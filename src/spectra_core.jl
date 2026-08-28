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

function replace_nonfinite_with_mean!(values; context="Spectrum")
    finite_sum = 0.0
    finite_count = 0
    @inbounds for value in values
        if isfinite(value)
            finite_sum += value
            finite_count += 1
        end
    end
    finite_count > 0 || error("$context requires finite values")
    replacement = Float32(finite_sum / finite_count)
    @inbounds for index in eachindex(values)
        isfinite(values[index]) || (values[index] = replacement)
    end
    return values
end

function prepare_spectrum_input!(workspace::SpectrumWorkspace, cube; remove_mean=true,
        window="none")
    size(workspace.input) == size(cube) || error("Spectrum workspace has the wrong size")
    @inbounds for index in eachindex(cube)
        workspace.input[index] = Float32(cube[index])
    end
    replace_nonfinite_with_mean!(workspace.input; context="Spectrum")
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
        replace_nonfinite_with_mean!(A; context="Vector spectrum")
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

function cross_spectrum_details(first, second; bins=80, box_size=(2π, 2π, 2π),
        axis_order=("x", "y", "z"))
    size(first) == size(second) || error("Cross-spectrum fields must have equal sizes")
    transformed = Array{ComplexF32,3}[]
    plan = plan_rfft(Array{Float32,3}(undef, size(first)); flags=FFTW.ESTIMATE)
    for source in (first, second)
        A = Float32.(source)
        replace_nonfinite_with_mean!(A; context="Cross spectrum")
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

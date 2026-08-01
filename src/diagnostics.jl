function periodic_index(index, offset, length)
    return mod1(index + offset, length)
end

function structure_functions(cube; orders=1:3, lags=nothing, samples_per_lag=100_000,
        seed=1234, periodic=true)
    selected_lags = isnothing(lags) ? unique!(round.(Int,
        10 .^ range(0, log10(max(1, minimum(size(cube)) ÷ 4)); length=16))) : Int.(lags)
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

function write_advanced_diagnostics!(files, fields, metadata, cfg, output_dir, overwrite)
    settings = get(cfg, "diagnostics", Dict())
    periodic = Bool(get(get(cfg, "grid", Dict()), "periodic", true))
    grid = get(cfg, "grid", Dict())
    box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    for name in string.(get(settings, "structure_fields", String[]))
        haskey(fields, name) || error("Structure function references missing field '$name'")
        lags, orders, values, counts = structure_functions(fields[name];
            orders=Int.(get(settings, "structure_orders", [1, 2, 3])),
            samples_per_lag=Int(get(settings, "structure_samples", 100_000)),
            seed=Int(get(settings, "seed", 1234)), periodic)
        path = joinpath(output_dir, "structure_$(sanitize(name)).csv")
        write_text_output(path; overwrite) do io
            println(io, join(["lag_cells", "samples", ["S$order" for order in orders]...], ','))
            for i in eachindex(lags)
                println(io, join([lags[i], counts[i], values[i, :]...], ','))
            end
        end
        push!(files, path)
    end
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
end

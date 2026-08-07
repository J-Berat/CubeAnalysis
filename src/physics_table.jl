struct WeightedStatistics
    samples::Int
    median::Float64
    mean::Float64
    deviation::Float64
    q16::Float64
    q84::Float64
end

const EMPTY_STATISTICS = WeightedStatistics(0, NaN, NaN, NaN, NaN, NaN)

"""Fluctuation amplitude of a vector field relative to its mean vector."""
struct FieldFluctuation
    mean_magnitude::Float64
    rms_fluctuation::Float64
    ratio::Float64
end

const EMPTY_FLUCTUATION = FieldFluctuation(NaN, NaN, NaN)

"""
Accumulates the weighted sums needed for the mean vector and the mean square
magnitude of a three-component field.
"""
mutable struct VectorAccumulator
    x::Float64
    y::Float64
    z::Float64
    square::Float64
    weight::Float64
end

VectorAccumulator() = VectorAccumulator(0.0, 0.0, 0.0, 0.0, 0.0)

@inline function accumulate_vector!(accumulator::VectorAccumulator, x, y, z, weight)
    accumulator.x += weight * x
    accumulator.y += weight * y
    accumulator.z += weight * z
    accumulator.square += weight * (x * x + y * y + z * z)
    accumulator.weight += weight
    return accumulator
end

"""
Turn accumulated sums into `|<F>|`, `sqrt(<|F|^2> - |<F>|^2)` and their ratio.
`scale` converts the cube values into the reported physical unit.
"""
function field_fluctuation(accumulator::VectorAccumulator, scale)
    accumulator.weight > 0 || return EMPTY_FLUCTUATION
    mx = accumulator.x / accumulator.weight
    my = accumulator.y / accumulator.weight
    mz = accumulator.z / accumulator.weight
    mean_magnitude = sqrt(mx^2 + my^2 + mz^2)
    mean_square = accumulator.square / accumulator.weight
    fluctuation = sqrt(max(mean_square - mean_magnitude^2, 0.0))
    return FieldFluctuation(scale * mean_magnitude, scale * fluctuation,
        mean_magnitude > 0 ? fluctuation / mean_magnitude : NaN)
end

"""
Quantiles of `values` under non-negative `weights`, for `probabilities` given in
increasing order. Uses the nearest-rank convention on the weighted cumulative
distribution.
"""
function weighted_quantiles(values::AbstractVector{Float64},
        weights::AbstractVector{Float64}, probabilities)
    length(values) == length(weights) ||
        error("Weighted quantiles need one weight per value")
    targets = Float64.(collect(probabilities))
    issorted(targets) || error("Weighted-quantile probabilities must increase")
    results = fill(NaN, length(targets))
    isempty(values) && return results
    total = 0.0
    @inbounds for weight in weights
        weight > 0 && (total += weight)
    end
    total > 0 || return results
    targets .*= total
    order = sortperm(values)
    cumulative = 0.0
    next = 1
    @inbounds for position in order
        weight = weights[position]
        weight > 0 || continue
        cumulative += weight
        while next <= length(targets) && cumulative >= targets[next]
            results[next] = values[position]
            next += 1
        end
        next > length(targets) && break
    end
    # Floating-point shortfall on the last bin: fall back to the largest value.
    @inbounds for index in next:length(targets)
        results[index] = values[last(order)]
    end
    return results
end

function weighted_statistics(values::AbstractVector{Float64},
        weights::AbstractVector{Float64})
    isempty(values) && return EMPTY_STATISTICS
    total = 0.0
    weighted_sum = 0.0
    samples = 0
    @inbounds for index in eachindex(values, weights)
        weight = weights[index]
        weight > 0 || continue
        total += weight
        weighted_sum += weight * values[index]
        samples += 1
    end
    total > 0 || return EMPTY_STATISTICS
    mean_value = weighted_sum / total
    variance = 0.0
    @inbounds for index in eachindex(values, weights)
        weight = weights[index]
        weight > 0 || continue
        variance += weight * (values[index] - mean_value)^2
    end
    q16, median_value, q84 = weighted_quantiles(values, weights, (0.16, 0.5, 0.84))
    return WeightedStatistics(samples, median_value, mean_value,
        sqrt(max(variance / total, 0.0)), q16, q84)
end

"""Unweighted (volume-weighted) counterpart of [`weighted_statistics`](@ref)."""
function uniform_statistics(values::AbstractVector{Float64})
    isempty(values) && return EMPTY_STATISTICS
    q16, median_value, q84 = quantile(values, (0.16, 0.5, 0.84))
    return WeightedStatistics(length(values), median_value, mean(values),
        std(values; corrected=false), q16, q84)
end

"""
Single pass over the strided cells producing the sonic and Alfvenic Mach
numbers, the plasma beta, and the turbulent-to-ordered ratios of the magnetic
and velocity fields, each volume-weighted and mass-weighted.

The sonic Mach number only requires a positive density and temperature, so it
keeps cells with a vanishing magnetic field that the Alfvenic Mach number and
the plasma beta must discard.
"""
function physics_summary(fields, spec; stride=1)
    density_name = string(get(spec, "density", "density"))
    temperature_name = string(get(spec, "temperature", "temperature"))
    velocity_names = string.(get(spec, "velocity", ["Vx", "Vy", "Vz"]))
    magnetic_names = string.(get(spec, "magnetic", ["Bx", "By", "Bz"]))
    length(velocity_names) == 3 || error("diagnostics.velocity needs three fields")
    length(magnetic_names) == 3 || error("diagnostics.magnetic needs three fields")
    names = [density_name; temperature_name; velocity_names; magnetic_names]
    all(haskey(fields, name) for name in names) ||
        error("Physics summary references missing fields: " *
            string(filter(name -> !haskey(fields, name), names)))

    gamma = Float64(get(spec, "gamma", 5 / 3))
    mu = Float64(get(spec, "mean_molecular_weight", 1.27))
    velocity_to_cms = Float64(get(spec, "velocity_to_cms", 1e5))
    magnetic_to_gauss = Float64(get(spec, "magnetic_to_gauss", 1e-3))
    k_B = 1.380649e-16
    m_H = 1.6735575e-24

    density = fields[density_name]
    temperature = fields[temperature_name]
    vx, vy, vz = (fields[name] for name in velocity_names)
    bx, by, bz = (fields[name] for name in magnetic_names)

    mach_sonic = Float64[]
    sonic_weights = Float64[]
    mach_alfven = Float64[]
    plasma_beta = Float64[]
    magnetized_weights = Float64[]
    expected = cld(length(density), stride)
    for buffer in (mach_sonic, sonic_weights, mach_alfven, plasma_beta, magnetized_weights)
        sizehint!(buffer, expected)
    end

    volume_magnetic = VectorAccumulator()
    mass_magnetic = VectorAccumulator()
    volume_velocity = VectorAccumulator()
    mass_velocity = VectorAccumulator()

    @inbounds for index in 1:stride:length(density)
        n = Float64(density[index])
        T = Float64(temperature[index])
        b1 = Float64(bx[index]); b2 = Float64(by[index]); b3 = Float64(bz[index])
        v1 = Float64(vx[index]); v2 = Float64(vy[index]); v3 = Float64(vz[index])
        mass = isfinite(n) && n > 0 ? n : 0.0
        magnetic_finite = isfinite(b1) && isfinite(b2) && isfinite(b3)
        velocity_finite = isfinite(v1) && isfinite(v2) && isfinite(v3)
        if magnetic_finite
            accumulate_vector!(volume_magnetic, b1, b2, b3, 1.0)
            mass > 0 && accumulate_vector!(mass_magnetic, b1, b2, b3, mass)
        end
        if velocity_finite
            accumulate_vector!(volume_velocity, v1, v2, v3, 1.0)
            mass > 0 && accumulate_vector!(mass_velocity, v1, v2, v3, mass)
        end
        mass > 0 && isfinite(T) && T > 0 && velocity_finite || continue
        speed = velocity_to_cms * sqrt(v1^2 + v2^2 + v3^2)
        sound = sqrt(gamma * k_B * T / (mu * m_H))
        push!(mach_sonic, speed / sound)
        push!(sonic_weights, mass)
        magnetic_finite || continue
        field = magnetic_to_gauss * sqrt(b1^2 + b2^2 + b3^2)
        field > 0 || continue
        rho = mu * m_H * n
        push!(mach_alfven, speed / (field / sqrt(4π * rho)))
        push!(plasma_beta, 8π * n * k_B * T / field^2)
        push!(magnetized_weights, mass)
    end

    distributions = [
        (key="mach_sonic", label="\\mathcal{M}_{s}",
            volume=uniform_statistics(mach_sonic),
            mass=weighted_statistics(mach_sonic, sonic_weights)),
        (key="mach_alfven", label="\\mathcal{M}_{A}",
            volume=uniform_statistics(mach_alfven),
            mass=weighted_statistics(mach_alfven, magnetized_weights)),
        (key="plasma_beta", label="\\beta",
            volume=uniform_statistics(plasma_beta),
            mass=weighted_statistics(plasma_beta, magnetized_weights)),
    ]
    magnetic_scale = magnetic_to_gauss * 1e6
    velocity_scale = velocity_to_cms / 1e5
    return (distributions=distributions,
        magnetic=(volume=field_fluctuation(volume_magnetic, magnetic_scale),
            mass=field_fluctuation(mass_magnetic, magnetic_scale)),
        velocity=(volume=field_fluctuation(volume_velocity, velocity_scale),
            mass=field_fluctuation(mass_velocity, velocity_scale)),
        magnetic_unit="\\mu\\mathrm{G}",
        velocity_unit="\\mathrm{km\\,s^{-1}}")
end

"""
LaTeX fragment (without delimiters) for one table cell. `digits` is lowered for
interval endpoints, which have to fit two numbers in one column.
"""
function table_number(value::Real; digits::Int=3)
    isfinite(value) || return "\\mathrm{n/a}"
    value == 0 && return "0"
    rendered = @sprintf("%.*g", digits, Float64(value))
    matched = match(r"^([+-]?[0-9.]+)e([+-]?[0-9]+)$", rendered)
    isnothing(matched) && return rendered
    return string(matched.captures[1], "\\times 10^{",
        parse(Int, matched.captures[2]), "}")
end

table_cell(value::Real) = latexstring(table_number(value))

# A hyphen inside math mode renders as a minus sign, so intervals use bracket
# notation rather than "a - b".
table_range(low::Real, high::Real) =
    latexstring("[", table_number(low; digits=2), ",\\ ",
        table_number(high; digits=2), "]")

const PHYSICS_TABLE_LABEL_X = 0.005
const PHYSICS_TABLE_RANGE_HEADER = latexstring("[q_{16},\\ q_{84}]")
# The mean and the standard deviation stay in physics_diagnostics.csv: the
# figure shows the median and the 16-84 % interval, which describe the heavy
# tails of Mach and beta distributions far better and always fit the column.
const PHYSICS_TABLE_COLUMNS = (volume=(median=0.255, range=0.565),
    mass=(median=0.700, range=1.000))
const PHYSICS_TABLE_GROUPS = (volume=(center=0.380, left=0.190, right=0.565),
    mass=(center=0.820, left=0.635, right=1.000))

function physics_table_rows(summary)
    rows = Any[(kind=:section, label="distributions per cell")]
    for entry in summary.distributions
        push!(rows, (kind=:distribution, label=entry.label,
            volume=entry.volume, mass=entry.mass))
    end
    push!(rows, (kind=:section, label="global field statistics"))
    magnetic_unit = summary.magnetic_unit
    velocity_unit = summary.velocity_unit
    push!(rows, (kind=:scalar, label="|\\langle B\\rangle|\\ [$magnetic_unit]",
        volume=summary.magnetic.volume.mean_magnitude,
        mass=summary.magnetic.mass.mean_magnitude))
    push!(rows, (kind=:scalar,
        label="\\delta B_{\\mathrm{rms}}\\ [$magnetic_unit]",
        volume=summary.magnetic.volume.rms_fluctuation,
        mass=summary.magnetic.mass.rms_fluctuation))
    push!(rows, (kind=:scalar,
        label="\\delta B_{\\mathrm{rms}}/|\\langle B\\rangle|",
        volume=summary.magnetic.volume.ratio, mass=summary.magnetic.mass.ratio))
    push!(rows, (kind=:scalar, label="|\\langle v\\rangle|\\ [$velocity_unit]",
        volume=summary.velocity.volume.mean_magnitude,
        mass=summary.velocity.mass.mean_magnitude))
    push!(rows, (kind=:scalar,
        label="\\delta v_{\\mathrm{rms}}\\ [$velocity_unit]",
        volume=summary.velocity.volume.rms_fluctuation,
        mass=summary.velocity.mass.rms_fluctuation))
    push!(rows, (kind=:scalar,
        label="\\delta v_{\\mathrm{rms}}/|\\langle v\\rangle|",
        volume=summary.velocity.volume.ratio, mass=summary.velocity.mass.ratio))
    return rows
end

function physics_table_figure(summary)
    rows = physics_table_rows(summary)
    top = 1.75
    bottom = -0.45 - 0.85 * (length(rows) - 1) - 0.7
    figure = publication_figure(size=(1400, round(Int, 118 + 38 * length(rows))),
        fontsize=17)
    axis = Axis(figure[1, 1]; backgroundcolor=:white,
        xautolimitmargin=(0.0, 0.0), yautolimitmargin=(0.0, 0.0))
    hidedecorations!(axis)
    hidespines!(axis)

    for (group, position) in pairs(PHYSICS_TABLE_GROUPS)
        stripes = group == :volume ? (PLOT_BLUE, 0.05) : (PLOT_ORANGE, 0.05)
        band!(axis, [position.left - 0.02, position.right + 0.012],
            fill(bottom + 0.35, 2), fill(top - 0.25, 2); color=stripes)
    end
    stripe = 0
    for (index, row) in enumerate(rows)
        row.kind == :section && continue
        stripe += 1
        isodd(stripe) || continue
        y = -0.45 - 0.85 * (index - 1)
        band!(axis, [-0.02, 1.02], fill(y - 0.42, 2), fill(y + 0.42, 2);
            color=(PLOT_MUTED, 0.07))
    end

    for (group, position) in pairs(PHYSICS_TABLE_GROUPS)
        text!(axis, position.center, 1.30;
            text=latex_text(group == :volume ? "volume weighted" : "mass weighted"),
            align=(:center, :center), color=PLOT_INK, fontsize=17)
        lines!(axis, [position.left, position.right], fill(1.05, 2);
            color=PLOT_INK, linewidth=1.1)
    end
    for (group, columns) in pairs(PHYSICS_TABLE_COLUMNS)
        for (name, x) in pairs(columns)
            label = name == :range ? PHYSICS_TABLE_RANGE_HEADER :
                latex_text(string(name))
            text!(axis, x, 0.55; text=label, align=(:right, :center),
                color=PLOT_MUTED, fontsize=15)
        end
    end
    lines!(axis, [-0.02, 1.02], fill(0.15, 2); color=PLOT_INK, linewidth=1.4)

    for (index, row) in enumerate(rows)
        y = -0.45 - 0.85 * (index - 1)
        if row.kind == :section
            text!(axis, PHYSICS_TABLE_LABEL_X, y; text=latex_text(row.label),
                align=(:left, :center), color=PLOT_MUTED, fontsize=15)
            continue
        end
        text!(axis, PHYSICS_TABLE_LABEL_X, y; text=latexstring(row.label),
            align=(:left, :center), color=PLOT_INK, fontsize=18)
        for group in (:volume, :mass)
            columns = getproperty(PHYSICS_TABLE_COLUMNS, group)
            value = getproperty(row, group)
            if row.kind == :scalar
                text!(axis, getproperty(PHYSICS_TABLE_GROUPS, group).center, y;
                    text=table_cell(value), align=(:center, :center),
                    color=PLOT_INK, fontsize=17)
                continue
            end
            text!(axis, columns.median, y; text=table_cell(value.median),
                align=(:right, :center), color=PLOT_INK, fontsize=17)
            text!(axis, columns.range, y; text=table_range(value.q16, value.q84),
                align=(:right, :center), color=PLOT_MUTED, fontsize=15)
        end
    end
    lines!(axis, [-0.02, 1.02], fill(bottom + 0.35, 2); color=PLOT_INK, linewidth=1.4)

    xlims!(axis, -0.03, 1.05)
    ylims!(axis, bottom, top)
    return figure
end

function write_physics_summary!(files, summary, output_dir; overwrite=true)
    path = joinpath(output_dir, "physics_diagnostics.csv")
    write_text_output(path; overwrite) do io
        println(io, "quantity,weighting,samples,median,mean,std,q16,q84")
        for entry in summary.distributions, weighting in (:volume, :mass)
            statistics = getproperty(entry, weighting)
            @printf(io, "%s,%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e\n", entry.key,
                weighting, statistics.samples, statistics.median, statistics.mean,
                statistics.deviation, statistics.q16, statistics.q84)
        end
    end
    push!(files, path)
    # The global ratios are single numbers, not distributions, so they get their
    # own table rather than being padded into the columns above.
    fluctuation_path = joinpath(output_dir, "field_fluctuations.csv")
    write_text_output(fluctuation_path; overwrite) do io
        println(io, "field,weighting,unit,mean_magnitude,rms_fluctuation,ratio")
        for (name, fluctuation, unit) in
                (("magnetic", summary.magnetic, "microG"),
                 ("velocity", summary.velocity, "km/s"))
            for weighting in (:volume, :mass)
                value = getproperty(fluctuation, weighting)
                @printf(io, "%s,%s,%s,%.8e,%.8e,%.8e\n", name, weighting, unit,
                    value.mean_magnitude, value.rms_fluctuation, value.ratio)
            end
        end
    end
    push!(files, fluctuation_path)
    return path
end

function physics_table_fields(cfg)
    settings = get(cfg, "diagnostics", Dict())
    return [string(get(settings, "density", "density"));
        string(get(settings, "temperature", "temperature"));
        string.(get(settings, "velocity", ["Vx", "Vy", "Vz"]));
        string.(get(settings, "magnetic", ["Bx", "By", "Bz"]))]
end

function plot_physics_table!(files, fields, cfg, output_dir, formats, overwrite)
    settings = get(cfg, "diagnostics", Dict())
    required = physics_table_fields(cfg)
    missing = filter(name -> !haskey(fields, name), required)
    if !isempty(missing)
        @info "Skipping the physics table; the cube lacks required fields" missing
        return files
    end
    stride = Int(get(cfg["run"], "sample_stride", 1))
    enforce_working_set(cfg, 8 * sizeof(Float32) *
        prod(analysis_field_size(fields, first(required))); context="physics table")
    summary = physics_summary(fields, settings; stride)
    save_figure!(files, physics_table_figure(summary), output_dir,
        "physics_table", formats; overwrite)
    save_data(cfg) && write_physics_summary!(files, summary, output_dir; overwrite)
    return files
end

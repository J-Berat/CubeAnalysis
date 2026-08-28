"""Density in thermal equilibrium for the Koyama--Inutsuka cooling law."""
function thermal_equilibrium_density(temperature::Real)
    temperature > 0 || return NaN
    cooling_over_heating = 1.0e7 * exp(-1.184e5 / (temperature + 1000.0)) +
        1.4e-2 * sqrt(temperature) * exp(-92.0 / temperature)
    return inv(cooling_over_heating)
end

function thermal_equilibrium_curve(nmin::Real, nmax::Real; samples=1200,
        temperature_range=(5.0, 1.0e6))
    samples >= 2 || error("Thermal-equilibrium curve needs at least two samples")
    tmin, tmax = Float64.(temperature_range)
    0 < tmin < tmax || error("thermal_temperature_range must contain two positive increasing values")
    temperatures = 10 .^ range(log10(tmin), log10(tmax); length=samples)
    densities = thermal_equilibrium_density.(temperatures)
    keep = isfinite.(densities) .& (densities .>= nmin) .& (densities .<= nmax)
    return densities[keep], temperatures[keep]
end

axis_values(values, logarithmic) = logarithmic ? log10.(values) : values

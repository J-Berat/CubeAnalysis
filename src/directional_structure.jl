function directional_structure_functions(components; lags=nothing, orders=1:4,
        box_size=1.0, axis_order=("x", "y", "z"), periodic=true)
    length(components) == 3 || error("Directional structure functions need three components")
    shape = size(first(components))
    all(size(component) == shape for component in components) ||
        error("Directional structure-function components must have equal sizes")
    selected_lags = isnothing(lags) ? unique!(round.(Int,
        10 .^ range(0, log10(max(1, minimum(shape) ÷ 4)); length=12))) : Int.(lags)
    selected_orders = sort!(unique!(Int.(orders)))
    all(>(0), selected_lags) || error("Directional lags must be positive")
    all(>(0), selected_orders) || error("Structure-function orders must be positive")
    spacing = grid_spacings(first(components), box_size; axis_order)
    rows = NamedTuple[]
    for axis in 1:3, lag in selected_lags
        longitudinal = zeros(Float64, length(selected_orders))
        transverse = zeros(Float64, length(selected_orders))
        full = zeros(Float64, length(selected_orders))
        signed2 = 0.0; signed3 = 0.0; signed4 = 0.0; count = 0
        @inbounds for k in axes(first(components), 3), j in axes(first(components), 2), i in axes(first(components), 1)
            coordinates = (i, j, k); target_coordinate = coordinates[axis] + lag
            if periodic
                target_coordinate = mod1(target_coordinate, shape[axis])
            elseif target_coordinate > shape[axis]
                continue
            end
            target = axis == 1 ? (target_coordinate, j, k) :
                axis == 2 ? (i, target_coordinate, k) : (i, j, target_coordinate)
            delta = ntuple(d -> Float64(components[d][target...] - components[d][i, j, k]), 3)
            all(isfinite, delta) || continue
            longitudinal_delta = delta[axis]
            transverse_delta = sqrt(sum(delta[d]^2 for d in 1:3 if d != axis))
            full_delta = sqrt(sum(abs2, delta))
            for (order_index, order) in enumerate(selected_orders)
                longitudinal[order_index] += abs(longitudinal_delta)^order
                transverse[order_index] += transverse_delta^order
                full[order_index] += full_delta^order
            end
            signed2 += longitudinal_delta^2; signed3 += longitudinal_delta^3
            signed4 += longitudinal_delta^4; count += 1
        end
        count > 0 || continue
        longitudinal ./= count; transverse ./= count; full ./= count
        variance = signed2 / count
        skewness = variance > 0 ? (signed3 / count) / variance^(3 / 2) : NaN
        flatness = variance > 0 ? (signed4 / count) / variance^2 : NaN
        push!(rows, (axis=string(axis_order[axis]), lag_cells=lag,
            separation=lag * spacing[axis], samples=count, orders=selected_orders,
            longitudinal, transverse, full, skewness, flatness))
    end
    return rows
end

function directional_scaling_exponents(rows, fit_range)
    axes = unique(row.axis for row in rows)
    orders = isempty(rows) ? Int[] : rows[1].orders
    fits = NamedTuple[]
    for axis in axes, (order_index, order) in enumerate(orders), component in (:longitudinal, :transverse, :full)
        selected = filter(row -> row.axis == axis && fit_range[1] <= row.separation <= fit_range[2], rows)
        separation = [row.separation for row in selected]
        moments = [getfield(row, component)[order_index] for row in selected]
        slope, stderr, points = spectral_fit(separation, moments, fit_range)
        push!(fits, (; axis, order, component=string(component), exponent=slope,
            exponent_std=stderr, points))
    end
    return fits
end

function plot_directional_structure!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict()); box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    periodic = Bool(get(grid, "periodic", true))
    for spec in get(cfg, "directional_structure_functions", Any[])
        name = string(spec["name"]); component_names = string.(spec["components"])
        length(component_names) == 3 || error("Directional structure '$name' needs three components")
        all(haskey(fields, field) for field in component_names) ||
            error("Directional structure '$name' references missing fields")
        components = [fields[field] for field in component_names]
        rows = directional_structure_functions(components;
            lags=get(spec, "lags", nothing), orders=Int.(get(spec, "orders", [1, 2, 3, 4])),
            box_size, axis_order, periodic)
        isempty(rows) && (@warn "No directional increments" name; continue)
        orders = rows[1].orders
        if save_data(cfg)
            path = joinpath(output_dir, "directional_structure_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                columns = ["axis", "lag_cells", "separation", "samples", "skewness", "flatness"]
                for order in orders, component in ("longitudinal", "transverse", "full")
                    push!(columns, "$(component)_S$(order)")
                end
                println(io, join(columns, ','))
                for row in rows
                    values = Any[row.axis, row.lag_cells, row.separation, row.samples, row.skewness, row.flatness]
                    for order_index in eachindex(orders)
                        append!(values, (row.longitudinal[order_index], row.transverse[order_index], row.full[order_index]))
                    end
                    println(io, join(values, ','))
                end
            end
            push!(files, path)
        end
        separations = getfield.(rows, :separation)
        fit_range = Float64.(get(spec, "fit_range", [minimum(separations), maximum(separations)]))
        fits = directional_scaling_exponents(rows, fit_range)
        if save_data(cfg)
            fit_path = joinpath(output_dir, "directional_structure_$(sanitize(name))_fits.csv")
            write_text_output(fit_path; overwrite) do io
                println(io, "axis,order,component,exponent,exponent_std,points")
                for fit in fits
                    @printf(io, "%s,%d,%s,%.8e,%.8e,%d\n", fit.axis, fit.order,
                        fit.component, fit.exponent, fit.exponent_std, fit.points)
                end
            end
            push!(files, fit_path)
        end
        second_index = findfirst(==(2), orders)
        if !isnothing(second_index)
            fig = publication_figure(size=(940, 600)); ax = latex_axis(fig[1, 1], xscale=log10, yscale=log10,
                xlabel="separation", ylabel="second-order increment", title=replace(name, '_' => ' '))
            for (axis_index, axis) in enumerate(axis_order)
                selected = filter(row -> row.axis == axis, rows)
                lines!(ax, getfield.(selected, :separation),
                    [row.longitudinal[second_index] for row in selected];
                    color=series_color(axis_index), linewidth=2.6,
                    label=latex_legend_label("L, $axis"))
                lines!(ax, getfield.(selected, :separation),
                    [row.transverse[second_index] for row in selected];
                    color=series_color(axis_index), linestyle=:dash, linewidth=2.2,
                    label=latex_legend_label("T, $axis"))
            end
            publication_legend!(ax)
            save_figure!(files, fig, output_dir, "directional_structure_$(sanitize(name))", formats; overwrite)
        end
    end
end

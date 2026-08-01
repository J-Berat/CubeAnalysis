function connected_excursion_metrics(active::BitArray{3}; periodic=true)
    visited = falses(size(active))
    linear = LinearIndices(active); cartesian = CartesianIndices(active)
    components = 0; largest = 0; spanning = falses(3)
    queue = Int[]
    for seed in eachindex(active)
        active[seed] && !visited[seed] || continue
        components += 1; empty!(queue); push!(queue, seed); visited[seed] = true
        head = 1; component_size = 0
        touches_low = falses(3); touches_high = falses(3)
        while head <= length(queue)
            index = queue[head]; head += 1; component_size += 1
            i, j, k = Tuple(cartesian[index]); coordinates = (i, j, k)
            for dimension in 1:3
                touches_low[dimension] |= coordinates[dimension] == 1
                touches_high[dimension] |= coordinates[dimension] == size(active, dimension)
                for offset in (-1, 1)
                    neighbor = coordinates[dimension] + offset
                    if periodic
                        neighbor = mod1(neighbor, size(active, dimension))
                    elseif !(1 <= neighbor <= size(active, dimension))
                        continue
                    end
                    target = dimension == 1 ? linear[neighbor, j, k] :
                        dimension == 2 ? linear[i, neighbor, k] : linear[i, j, neighbor]
                    if active[target] && !visited[target]
                        visited[target] = true; push!(queue, target)
                    end
                end
            end
        end
        largest = max(largest, component_size)
        spanning .|= touches_low .& touches_high
    end
    return components, largest, Tuple(spanning)
end

function excursion_metrics(field, threshold; above=true, periodic=true, box_size=1.0,
        axis_order=("x", "y", "z"))
    active = BitArray(undef, size(field))
    @inbounds for index in eachindex(field)
        value = Float64(field[index])
        active[index] = isfinite(value) && (above ? value >= threshold : value <= threshold)
    end
    active_count = count(active); total = length(active)
    active_count == 0 && return (threshold=Float64(threshold), volume_fraction=0.0,
        surface_area=0.0, surface_density=0.0, components=0, largest_fraction=0.0,
        spans=(false, false, false))
    lengths = box_lengths(box_size, axis_order)
    spacing = ntuple(d -> lengths[d] / size(field, d), 3)
    face_areas = ntuple(d -> prod(spacing[q] for q in 1:3 if q != d), 3)
    surface_area = 0.0
    @inbounds for k in axes(active, 3), j in axes(active, 2), i in axes(active, 1)
        cell_active = active[i, j, k]
        coordinates = (i, j, k)
        for dimension in 1:3
            next_coordinate = coordinates[dimension] + 1
            if next_coordinate <= size(active, dimension)
                neighbor_active = dimension == 1 ? active[next_coordinate, j, k] :
                    dimension == 2 ? active[i, next_coordinate, k] : active[i, j, next_coordinate]
                neighbor_active != cell_active && (surface_area += face_areas[dimension])
            elseif periodic
                neighbor_active = dimension == 1 ? active[1, j, k] :
                    dimension == 2 ? active[i, 1, k] : active[i, j, 1]
                neighbor_active != cell_active && (surface_area += face_areas[dimension])
            elseif cell_active
                surface_area += face_areas[dimension]
            end
            if !periodic && cell_active && coordinates[dimension] == 1
                surface_area += face_areas[dimension]
            end
        end
    end
    components, largest, spans = connected_excursion_metrics(active; periodic)
    volume = prod(lengths)
    return (threshold=Float64(threshold), volume_fraction=active_count / total,
        surface_area, surface_density=surface_area / volume, components,
        largest_fraction=largest / active_count, spans)
end

function plot_topology!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict()); box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    periodic = Bool(get(grid, "periodic", true))
    for spec in get(cfg, "topology", Any[])
        name = string(get(spec, "name", spec["field"])); field_name = string(spec["field"])
        haskey(fields, field_name) || error("Topology '$name' references missing field '$field_name'")
        field = fields[field_name]
        enforce_working_set(cfg, 14 * length(field); context="excursion topology '$name'")
        thresholds = if haskey(spec, "thresholds")
            Float64.(spec["thresholds"])
        else
            quantiles = Float64.(get(spec, "quantiles", collect(0.1:0.1:0.9)))
            all(q -> 0 <= q <= 1, quantiles) || error("Topology quantiles must lie in [0,1]")
            sample = sampled_field(fields, field_name;
                stride=Int(get(cfg["run"], "sample_stride", 1)))
            quantile(sample, quantiles)
        end
        results = [excursion_metrics(field, threshold;
            above=Bool(get(spec, "above", true)), periodic, box_size, axis_order)
            for threshold in thresholds]
        path = joinpath(output_dir, "topology_$(sanitize(name)).csv")
        write_text_output(path; overwrite) do io
            println(io, "threshold,volume_fraction,surface_area,surface_density,components,largest_fraction,spans_x,spans_y,spans_z")
            for result in results
                @printf(io, "%.8e,%.8e,%.8e,%.8e,%d,%.8e,%s,%s,%s\n", result.threshold,
                    result.volume_fraction, result.surface_area, result.surface_density,
                    result.components, result.largest_fraction, result.spans...)
            end
        end
        push!(files, path)
        fig = Figure(size=(1050, 480))
        ax1 = Axis(fig[1, 1], xlabel=field_label(field_name, metadata[field_name]),
            ylabel="fraction", title="Excursion-set morphology")
        lines!(ax1, thresholds, getfield.(results, :volume_fraction); label="volume", linewidth=2.5)
        lines!(ax1, thresholds, getfield.(results, :largest_fraction); label="largest component", linewidth=2.5)
        axislegend(ax1)
        ax2 = Axis(fig[1, 2], xlabel=field_label(field_name, metadata[field_name]),
            ylabel="connected components")
        lines!(ax2, thresholds, getfield.(results, :components); linewidth=2.5, color=:darkorange)
        save_figure!(files, fig, output_dir, "topology_$(sanitize(name))", formats; overwrite)
    end
end

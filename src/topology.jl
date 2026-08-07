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

"""
Count the cells of the cubical complex spanned by the active voxels: vertices,
edges (per direction), faces (per normal direction) and cubes.

Every intrinsic volume of a voxel set follows from these four counts. On a unit
cube (n0, n1, n2, n3) = (8, 12, 6, 1) gives volume 1, surface 6, V1 = 3 and
Euler characteristic 1.
"""
function cubical_complex_counts(active::BitArray{3}; periodic=true)
    shape = size(active)
    extent = periodic ? shape : shape .+ 1
    wrap(value, dimension) = periodic ? mod1(value, shape[dimension]) : value
    vertices = falses(extent)
    edges = ntuple(_ -> falses(extent), 3)
    faces = ntuple(_ -> falses(extent), 3)
    @inbounds for k in axes(active, 3), j in axes(active, 2), i in axes(active, 1)
        active[i, j, k] || continue
        for dk in 0:1, dj in 0:1, di in 0:1
            vertices[wrap(i + di, 1), wrap(j + dj, 2), wrap(k + dk, 3)] = true
        end
        for dk in 0:1, dj in 0:1
            edges[1][i, wrap(j + dj, 2), wrap(k + dk, 3)] = true
        end
        for dk in 0:1, di in 0:1
            edges[2][wrap(i + di, 1), j, wrap(k + dk, 3)] = true
        end
        for dj in 0:1, di in 0:1
            edges[3][wrap(i + di, 1), wrap(j + dj, 2), k] = true
        end
        for di in 0:1
            faces[1][wrap(i + di, 1), j, k] = true
        end
        for dj in 0:1
            faces[2][i, wrap(j + dj, 2), k] = true
        end
        for dk in 0:1
            faces[3][i, j, wrap(k + dk, 3)] = true
        end
    end
    return (vertices=count(vertices), edges=Tuple(count(e) for e in edges),
        faces=Tuple(count(f) for f in faces), cubes=count(active))
end

"""
The four Minkowski functionals of the excursion set.

`euler` is the Euler characteristic `n0 - n1 + n2 - n3`, and `mean_width` the
first intrinsic volume `V1`, which for an `a x b x c` box equals `a + b + c`.
`V1` is only reported in physical units on an isotropic grid; otherwise it is
given in cells.
"""
function minkowski_functionals(counts, spacing)
    n0 = counts.vertices
    n1 = sum(counts.edges)
    n2 = sum(counts.faces)
    n3 = counts.cubes
    euler = n0 - n1 + n2 - n3
    isotropic = maximum(spacing) / minimum(spacing) <= 1 + 1e-8
    cell = isotropic ? first(spacing) : 1.0
    mean_width = cell * (n1 - 2 * n2 + 3 * n3)
    return (euler=euler, mean_width=mean_width, isotropic=isotropic)
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
        spans=(false, false, false), euler=0, euler_density=0.0, mean_width=0.0,
        mean_width_isotropic=true)
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
    minkowski = minkowski_functionals(
        cubical_complex_counts(active; periodic), spacing)
    volume = prod(lengths)
    return (threshold=Float64(threshold), volume_fraction=active_count / total,
        surface_area, surface_density=surface_area / volume, components,
        largest_fraction=largest / active_count, spans,
        euler=minkowski.euler, euler_density=minkowski.euler / volume,
        mean_width=minkowski.mean_width,
        mean_width_isotropic=minkowski.isotropic)
end

function plot_topology!(files, fields, metadata, cfg, output_dir, formats, overwrite)
    grid = get(cfg, "grid", Dict()); box_size = get(grid, "box_size", 1.0)
    axis_order = string.(get(grid, "axis_order", ["x", "y", "z"]))
    periodic = Bool(get(grid, "periodic", true))
    for spec in get(cfg, "topology", Any[])
        name = string(get(spec, "name", spec["field"])); field_name = string(spec["field"])
        haskey(fields, field_name) || error("Topology '$name' references missing field '$field_name'")
        field = fields[field_name]
        # Each thread builds its own active/visited bitmaps and BFS queue.
        enforce_working_set(cfg, 14 * length(field) * Threads.nthreads();
            context="excursion topology '$name'")
        thresholds = if haskey(spec, "thresholds")
            Float64.(spec["thresholds"])
        else
            quantiles = Float64.(get(spec, "quantiles", collect(0.1:0.1:0.9)))
            all(q -> 0 <= q <= 1, quantiles) || error("Topology quantiles must lie in [0,1]")
            sample = sampled_field(fields, field_name;
                stride=Int(get(cfg["run"], "sample_stride", 1)))
            quantile(sample, quantiles)
        end
        above = Bool(get(spec, "above", true))
        results = Vector{Any}(undef, length(thresholds))
        Threads.@threads for index in eachindex(thresholds)
            results[index] = excursion_metrics(field, thresholds[index];
                above, periodic, box_size, axis_order)
        end
        results = [result for result in results]
        if save_data(cfg)
            path = joinpath(output_dir, "topology_$(sanitize(name)).csv")
            write_text_output(path; overwrite) do io
                println(io, "threshold,volume_fraction,surface_area,surface_density,mean_width,euler,euler_density,components,largest_fraction,spans_x,spans_y,spans_z")
                for result in results
                    @printf(io, "%.8e,%.8e,%.8e,%.8e,%.8e,%d,%.8e,%d,%.8e,%s,%s,%s\n",
                        result.threshold, result.volume_fraction, result.surface_area,
                        result.surface_density, result.mean_width, result.euler,
                        result.euler_density, result.components,
                        result.largest_fraction, result.spans...)
                end
            end
            push!(files, path)
        end
        threshold_label = field_label(field_name, metadata[field_name])
        volume = prod(box_lengths(box_size, axis_order))
        isotropic = all(result -> result.mean_width_isotropic, results)
        width_unit = isotropic ? "" : " [cells]"
        # The four Minkowski functionals: volume, surface, mean width and the
        # Euler characteristic, each normalised by the box volume.
        panels = (
            (latex_text("volume fraction"), getfield.(results, :volume_fraction), PLOT_BLUE),
            (latex_text("surface density"), getfield.(results, :surface_density), PLOT_ORANGE),
            (latex_text("mean width density$width_unit"),
                getfield.(results, :mean_width) ./ volume, PLOT_TEAL),
            (latexstring("\\chi/V"), getfield.(results, :euler_density), PLOT_PURPLE),
        )
        fig = publication_figure(size=(1500, 780))
        for (index, (ylabel, values, color)) in enumerate(panels)
            row, column = fldmod1(index, 2)
            ax = latex_axis(fig[row, column], xlabel=threshold_label, ylabel=ylabel)
            index == 4 && hlines!(ax, [0.0]; color=PLOT_MUTED, linestyle=:dash,
                linewidth=1.5)
            lines!(ax, thresholds, values; linewidth=2.7, color)
            scatter!(ax, thresholds, values; color=:white, strokecolor=color,
                strokewidth=1.3, markersize=7)
        end
        colgap!(fig.layout, 40); rowgap!(fig.layout, 30)
        save_figure!(files, fig, output_dir, "minkowski_$(sanitize(name))", formats;
            overwrite)

        fig = publication_figure(size=(1100, 520))
        ax1 = latex_axis(fig[1, 1], xlabel=threshold_label, ylabel="fraction")
        lines!(ax1, thresholds, getfield.(results, :volume_fraction);
            label=latex_legend_label("volume"), linewidth=2.7, color=PLOT_BLUE)
        lines!(ax1, thresholds, getfield.(results, :largest_fraction);
            label=latex_legend_label("largest component"), linewidth=2.5,
            color=PLOT_ORANGE, linestyle=:dash)
        publication_legend!(ax1)
        ax2 = latex_axis(fig[1, 2], xlabel=threshold_label,
            ylabel="connected components")
        lines!(ax2, thresholds, getfield.(results, :components); linewidth=2.7,
            color=PLOT_TEAL)
        save_figure!(files, fig, output_dir, "topology_$(sanitize(name))", formats; overwrite)
    end
end

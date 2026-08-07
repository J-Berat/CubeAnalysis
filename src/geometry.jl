"""Return physical box lengths in array-dimension order."""
function box_lengths(box_size, axis_order=("x", "y", "z"))
    order = lowercase.(string.(axis_order))
    length(order) == 3 && Set(order) == Set(("x", "y", "z")) ||
        error("grid.axis_order must contain x, y, and z exactly once")
    physical = if box_size isa Real
        fill(Float64(box_size), 3)
    else
        values = Float64.(box_size)
        length(values) == 3 || error("grid.box_size must be a scalar or three values")
        values
    end
    all(isfinite, physical) && all(>(0), physical) ||
        error("grid.box_size values must be positive and finite")
    canonical = Dict("x" => physical[1], "y" => physical[2], "z" => physical[3])
    return Tuple(canonical[name] for name in order)
end

grid_box_lengths(cfg) = begin
    grid = get(cfg, "grid", Dict())
    box_lengths(get(grid, "box_size", 1.0), get(grid, "axis_order", ["x", "y", "z"]))
end

function grid_spacings(A, box_size=1.0; axis_order=("x", "y", "z"))
    lengths = box_lengths(box_size, axis_order)
    return ntuple(d -> lengths[d] / size(A, d), 3)
end

function periodic_gradient(A, box_size=1.0; axis_order=("x", "y", "z"))
    dx, dy, dz = grid_spacings(A, box_size; axis_order)
    gx = (circshift(A, (-1, 0, 0)) .- circshift(A, (1, 0, 0))) ./ (2dx)
    gy = (circshift(A, (0, -1, 0)) .- circshift(A, (0, 1, 0))) ./ (2dy)
    gz = (circshift(A, (0, 0, -1)) .- circshift(A, (0, 0, 1))) ./ (2dz)
    return gx, gy, gz
end

function nonperiodic_gradient(A, box_size=1.0; axis_order=("x", "y", "z"))
    dx, dy, dz = grid_spacings(A, box_size; axis_order)
    gx, gy, gz = similar(A), similar(A), similar(A)
    @views begin
        gx[2:end-1, :, :] .= (A[3:end, :, :] .- A[1:end-2, :, :]) ./ (2dx)
        gx[1, :, :] .= (A[2, :, :] .- A[1, :, :]) ./ dx
        gx[end, :, :] .= (A[end, :, :] .- A[end-1, :, :]) ./ dx
        gy[:, 2:end-1, :] .= (A[:, 3:end, :] .- A[:, 1:end-2, :]) ./ (2dy)
        gy[:, 1, :] .= (A[:, 2, :] .- A[:, 1, :]) ./ dy
        gy[:, end, :] .= (A[:, end, :] .- A[:, end-1, :]) ./ dy
        gz[:, :, 2:end-1] .= (A[:, :, 3:end] .- A[:, :, 1:end-2]) ./ (2dz)
        gz[:, :, 1] .= (A[:, :, 2] .- A[:, :, 1]) ./ dz
        gz[:, :, end] .= (A[:, :, end] .- A[:, :, end-1]) ./ dz
    end
    return gx, gy, gz
end

@inline function derivative_at(A, i, j, k, dimension, spacing, periodic)
    coordinate = (i, j, k)[dimension]
    extent = size(A, dimension)
    if periodic
        lower = mod1(coordinate - 1, extent); upper = mod1(coordinate + 1, extent)
        if dimension == 1
            return (A[upper, j, k] - A[lower, j, k]) / (2spacing)
        elseif dimension == 2
            return (A[i, upper, k] - A[i, lower, k]) / (2spacing)
        else
            return (A[i, j, upper] - A[i, j, lower]) / (2spacing)
        end
    end
    lower = max(1, coordinate - 1); upper = min(extent, coordinate + 1)
    denominator = (upper - lower) * spacing
    if dimension == 1
        return (A[upper, j, k] - A[lower, j, k]) / denominator
    elseif dimension == 2
        return (A[i, upper, k] - A[i, lower, k]) / denominator
    else
        return (A[i, j, upper] - A[i, j, lower]) / denominator
    end
end

@inline function gradient_at(A, i, j, k, spacings, periodic)
    return (derivative_at(A, i, j, k, 1, spacings[1], periodic),
        derivative_at(A, i, j, k, 2, spacings[2], periodic),
        derivative_at(A, i, j, k, 3, spacings[3], periodic))
end

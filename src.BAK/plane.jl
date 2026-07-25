module plane

import Base: size, axes
import LinearAlgebra: norm, dot, cross

export Plane, PointIterator, XwiseIterator, YwiseIterator,
       to_plane, from_plane, from_plane2d,
       point_iterator, xwise_iterator, ywise_iterator

"""
    Plane{T<:Real}

3D plane with deterministic no-roll local coordinates.

The origin is a 3-vector. `orientation=(theta, phi)` defines the normal as
`[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]`. Since two angles
do not define roll about that normal, `ort_x` is chosen by projecting global
+x onto the plane, falling back to global +y if needed; `ort_y = cross(n, ort_x)`.
Plane dimensions are two-component `[span_x, span_y]` values.
Grid coordinates are produced on-demand by iterators.
"""
struct Plane{T<:Real}
    origin::Vector{T}
    span::Vector{T}
    resolution::Vector{T}
    orientation::Vector{T}
    num_nodes::Vector{Int}

    # Reference vectors
    ort_x::Vector{T}
    ort_y::Vector{T}
    ort_norm::Vector{T}

    # Scalar spacings
    dx_::T
    dy_::T

    function Plane{T}(origin,
                        span=nothing,
                        num_nodes=nothing,
                        resolution=nothing,
                        orientation=nothing) where {T<:Real}
        origin_v = convert(Vector{T}, collect(origin))
        length(origin_v) == 3 || throw(ArgumentError("origin must be a 3-vector"))

        orientation_v = orientation === nothing ? T[0, 0] : convert(Vector{T}, collect(orientation))
        length(orientation_v) == 2 || throw(ArgumentError("orientation must be [theta, phi]"))

        # Resolve span / resolution / num_nodes
        if span !== nothing && num_nodes !== nothing
            span_v = convert(Vector{T}, collect(span))
            length(span_v) == 2 || throw(ArgumentError("span must be a 2-vector"))
            any(isapprox.(span_v, 0)) && throw(ArgumentError("span cannot be zero"))
            nodes_v = convert(Vector{Int}, collect(num_nodes))
            length(nodes_v) == 2 || throw(ArgumentError("num_nodes must be a 2-vector"))
            any(nodes_v .< 1) && throw(ArgumentError("num_nodes must be >= 1"))
            res_v = T[ nodes_v[i] == 1 ? span_v[i] : span_v[i] / (nodes_v[i] - 1) for i in 1:2 ]
        elseif span !== nothing && resolution !== nothing
            span_v = convert(Vector{T}, collect(span))
            length(span_v) == 2 || throw(ArgumentError("span must be a 2-vector"))
            any(isapprox.(span_v, 0)) && throw(ArgumentError("span cannot be zero"))
            res_v = convert(Vector{T}, collect(resolution))
            length(res_v) == 2 || throw(ArgumentError("resolution must be a 2-vector"))
            any(isapprox.(res_v, 0)) && throw(ArgumentError("resolution cannot be zero"))
            nodes_v = Int.(floor.(span_v ./ res_v)) .+ 1
            any(nodes_v .< 1) && throw(ArgumentError("derived num_nodes must be >= 1"))
            res_v = T[ nodes_v[i] == 1 ? span_v[i] : span_v[i] / (nodes_v[i] - 1) for i in 1:2 ]
        elseif num_nodes !== nothing && resolution !== nothing
            nodes_v = convert(Vector{Int}, collect(num_nodes))
            length(nodes_v) == 2 || throw(ArgumentError("num_nodes must be a 2-vector"))
            any(nodes_v .< 1) && throw(ArgumentError("num_nodes must be >= 1"))
            res_v = convert(Vector{T}, collect(resolution))
            length(res_v) == 2 || throw(ArgumentError("resolution must be a 2-vector"))
            any(isapprox.(res_v, 0)) && throw(ArgumentError("resolution cannot be zero"))
            span_v = res_v .* (nodes_v .- 1)
            for i in 1:2
                if nodes_v[i] == 1
                    span_v[i] = res_v[i]
                end
            end
        else
            throw(ArgumentError("Must provide exactly two of: span, num_nodes, resolution"))
        end

        # Reference vectors: deterministic no-roll basis
        theta, phi = orientation_v
        ort_norm = T[sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
        ort_norm ./= norm(ort_norm)

        seed = T[1, 0, 0]
        if abs(dot(seed, ort_norm)) > T(0.95)
            seed = T[0, 1, 0]
        end
        ort_x = seed .- dot(seed, ort_norm) .* ort_norm
        ort_x ./= norm(ort_x)
        ort_y = cross(ort_norm, ort_x)
        ort_y ./= norm(ort_y)

        new{T}(origin_v, span_v, res_v, orientation_v, nodes_v,
               ort_x, ort_y, ort_norm,
               T(res_v[1]), T(res_v[2]))
    end
end

# Convenience outer constructor
Plane(origin; span=nothing, num_nodes=nothing, resolution=nothing,
        orientation=nothing, dtype::Type{T}=Float64) where {T<:Real} =
    Plane{T}(origin, span, num_nodes, resolution, orientation)


# ------------------------------------------------------------------
# Coordinate transforms
# ------------------------------------------------------------------

"""
    to_plane(pl::Plane, p::AbstractArray{T,3}) -> Array{T,3}

Transforms global points `(..., 3)` to local `(x_, y_, z_)`.
"""
function to_plane(pl::Plane, p::AbstractArray{T,3}) where {T}
    size(p, 3) == 3 || throw(ArgumentError("Last dimension must be 3"))
    p0 = p .- reshape(pl.origin, (1, 1, 3))
    x_ = sum(p0 .* reshape(pl.ort_x, (1, 1, 3)); dims=3)
    y_ = sum(p0 .* reshape(pl.ort_y, (1, 1, 3)); dims=3)
    z_ = sum(p0 .* reshape(pl.ort_norm, (1, 1, 3)); dims=3)
    cat(x_, y_, z_; dims=3)
end

function to_plane(pl::Plane, p::AbstractVector{T}) where {T}
    length(p) == 3 || throw(ArgumentError("Vector must have length 3"))
    p0 = p .- pl.origin
    return T[dot(p0, pl.ort_x), dot(p0, pl.ort_y), dot(p0, pl.ort_norm)]
end

function to_plane(pl::Plane, p::AbstractMatrix{T}) where {T}
    size(p, 2) == 3 || throw(ArgumentError("Matrix must have shape (N, 3)"))
    out = similar(p)
    for i in axes(p, 1)
        p0 = p[i, :] .- pl.origin
        out[i, 1] = dot(p0, pl.ort_x)
        out[i, 2] = dot(p0, pl.ort_y)
        out[i, 3] = dot(p0, pl.ort_norm)
    end
    return out
end

"""
    from_plane(pl::Plane, p_::AbstractArray{T,3}) -> Array{T,3}

Transforms local points `(..., 3)` to global coordinates.
"""
function from_plane(pl::Plane, p_::AbstractArray{T,3}) where {T}
    size(p_, 3) == 3 || throw(ArgumentError("Last dimension must be 3"))
    K = vcat(pl.ort_x', pl.ort_y', pl.ort_norm')
    p = reshape(p_, (:, 3)) * K
    p .+= pl.origin'
    reshape(p, size(p_))
end

function from_plane(pl::Plane, p_::AbstractVector{T}) where {T}
    length(p_) == 3 || throw(ArgumentError("Vector must have length 3"))
    return pl.origin .+ p_[1].*pl.ort_x .+ p_[2].*pl.ort_y .+ p_[3].*pl.ort_norm
end

function from_plane(pl::Plane, p_::AbstractMatrix{T}) where {T}
    size(p_, 2) == 3 || throw(ArgumentError("Matrix must have shape (N, 3)"))
    out = similar(p_)
    for i in axes(p_, 1)
        out[i, :] = pl.origin .+ p_[i,1].*pl.ort_x .+ p_[i,2].*pl.ort_y .+ p_[i,3].*pl.ort_norm
    end
    return out
end

"""
    from_plane2d(pl::Plane, p_::AbstractArray{T,N}) where {N}

Transforms local in-plane points `(..., 2)` to global 3D coordinates `(x, y, z)`.
"""
function from_plane2d(pl::Plane, p_::AbstractMatrix{T}) where {T}
    size(p_, 2) == 2 || throw(ArgumentError("Matrix must have shape (N, 2)"))
    out = similar(p_, T, (size(p_, 1), 3))
    for i in axes(p_, 1)
        out[i, :] = pl.origin .+ p_[i,1].*pl.ort_x .+ p_[i,2].*pl.ort_y
    end
    return out
end

function from_plane2d(pl::Plane, p_::AbstractVector{T}) where {T}
    length(p_) == 2 || throw(ArgumentError("Vector must have length 2"))
    return pl.origin .+ p_[1].*pl.ort_x .+ p_[2].*pl.ort_y
end


# ------------------------------------------------------------------
# Grid helpers (on-demand, no stored arrays)
# ------------------------------------------------------------------

@inline function _x_local(pl::Plane, i::Int)
    return pl.num_nodes[1] > 1 ? -pl.span[1]/2 + (i - 1) * pl.dx_ : zero(pl.dx_)
end

@inline function _y_local(pl::Plane, j::Int)
    return pl.num_nodes[2] > 1 ? -pl.span[2]/2 + (j - 1) * pl.dy_ : zero(pl.dy_)
end

@inline function _global_point(pl::Plane, xl::Real, yl::Real)
    return pl.origin .+ xl .* pl.ort_x .+ yl .* pl.ort_y
end


# ------------------------------------------------------------------
# Iterators
# ------------------------------------------------------------------

"""
    PointIterator{T}

Yields `(x, y, z)` each a `Vector{T}` of length 1, scanning the plane
point-by-point (row-major order: x fastest, then y).
"""
struct PointIterator{T<:Real}
    plane::Plane{T}
end

point_iterator(pl::Plane{T}) where {T} = PointIterator{T}(pl)

Base.length(it::PointIterator) = prod(it.plane.num_nodes)

function Base.iterate(it::PointIterator, state=(1, 1))
    nx, ny = it.plane.num_nodes
    i, j = state
    if j > ny
        return nothing
    end
    xl = _x_local(it.plane, i)
    yl = _y_local(it.plane, j)
    g = _global_point(it.plane, xl, yl)
    item = (g[1:1], g[2:2], g[3:3])
    if i < nx
        return item, (i + 1, j)
    else
        return item, (1, j + 1)
    end
end

Base.eltype(::Type{PointIterator{T}}) where {T} = Tuple{Vector{T}, Vector{T}, Vector{T}}


"""
    XwiseIterator{T}

Yields `(x, y, z)` each a `Matrix{T}` of size `(nx, sy)`, scanning the
plane x-wise with a y-stride of `sy`.
"""
struct XwiseIterator{T<:Real}
    plane::Plane{T}
    sy::Int
end

function xwise_iterator(pl::Plane{T}, sy::Int) where {T}
    sy <= pl.num_nodes[2] || throw(ArgumentError("sy must be <= ny=$(pl.num_nodes[2])"))
    sy > 0 || throw(ArgumentError("sy must be positive"))
    return XwiseIterator{T}(pl, sy)
end

function Base.length(it::XwiseIterator)
    ny = it.plane.num_nodes[2]
    return ceil(Int, ny / it.sy)
end

function Base.iterate(it::XwiseIterator, state=1)
    nx, ny = it.plane.num_nodes
    y_start = state
    if y_start > ny
        return nothing
    end
    y_end = min(y_start + it.sy - 1, ny)
    sy_eff = y_end - y_start + 1

    xs = similar(it.plane.origin, nx, sy_eff)
    ys = similar(it.plane.origin, nx, sy_eff)
    zs = similar(it.plane.origin, nx, sy_eff)

    pl = it.plane
    @inbounds for k in 1:sy_eff
        yl = _y_local(pl, y_start + k - 1)
        for i in 1:nx
            xl = _x_local(pl, i)
            g = _global_point(pl, xl, yl)
            xs[i, k] = g[1]
            ys[i, k] = g[2]
            zs[i, k] = g[3]
        end
    end

    return (xs, ys, zs), y_end + 1
end

Base.eltype(::Type{XwiseIterator{T}}) where {T} = Tuple{Matrix{T}, Matrix{T}, Matrix{T}}


"""
    YwiseIterator{T}

Yields `(x, y, z)` each a `Matrix{T}` of size `(sx, ny)`, scanning the
plane y-wise with an x-stride of `sx`.
"""
struct YwiseIterator{T<:Real}
    plane::Plane{T}
    sx::Int
end

function ywise_iterator(pl::Plane{T}, sx::Int) where {T}
    sx < pl.num_nodes[1] || throw(ArgumentError("sx must be < nx=$(pl.num_nodes[1])"))
    sx > 0 || throw(ArgumentError("sx must be positive"))
    return YwiseIterator{T}(pl, sx)
end

function Base.length(it::YwiseIterator)
    nx = it.plane.num_nodes[1]
    return ceil(Int, nx / it.sx)
end

function Base.iterate(it::YwiseIterator, state=1)
    nx, ny = it.plane.num_nodes
    x_start = state
    if x_start > nx
        return nothing
    end
    x_end = min(x_start + it.sx - 1, nx)
    sx_eff = x_end - x_start + 1

    xs = similar(it.plane.origin, sx_eff, ny)
    ys = similar(it.plane.origin, sx_eff, ny)
    zs = similar(it.plane.origin, sx_eff, ny)

    pl = it.plane
    @inbounds for j in 1:ny
        yl = _y_local(pl, j)
        for k in 1:sx_eff
            xl = _x_local(pl, x_start + k - 1)
            g = _global_point(pl, xl, yl)
            xs[k, j] = g[1]
            ys[k, j] = g[2]
            zs[k, j] = g[3]
        end
    end

    return (xs, ys, zs), x_end + 1
end

Base.eltype(::Type{YwiseIterator{T}}) where {T} = Tuple{Matrix{T}, Matrix{T}, Matrix{T}}


end # module plane

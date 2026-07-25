module fourier

import Base: size, axes
import FourierTools: czt, iczt

export direct_ft2, czt_ft2, iczt_ft2

function direct_ft2(f::AbstractMatrix{<:Number},
                    x_::AbstractVector{<:Real},
                    y_::AbstractVector{<:Real},
                    xi::AbstractVector{<:Real},
                    eta::AbstractVector{<:Real})
    ny, nx = size(f)
    length(x_) == nx || throw(ArgumentError("length(x_) must match size(f,2)"))
    length(y_) == ny || throw(ArgumentError("length(y_) must match size(f,1)"))

    nxi  = length(xi)
    neta = length(eta)

    dx_ = length(x_) > 1 ? abs(x_[2] - x_[1]) : one(eltype(x_))
    dy_ = length(y_) > 1 ? abs(y_[2] - y_[1]) : one(eltype(y_))

    phase_x = cis.(-2π .* xi .* x_')
    phase_y = cis.(-2π .* eta .* y_')

    g = f * phase_x'
    F = phase_y * g

    return dx_ * dy_ * F
end

function czt_ft2(f::AbstractMatrix{<:Number},
                 x_::AbstractVector{<:Real},
                 y_::AbstractVector{<:Real},
                 nxi::Int,
                 neta::Int;
                 dx::Union{Real,Nothing}=nothing,
                 dy::Union{Real,Nothing}=nothing)
    ny, nx = size(f)
    length(x_) == nx || throw(ArgumentError("length(x_) must match size(f,2)"))
    length(y_) == ny || throw(ArgumentError("length(y_) must match size(f,1)"))

    dx_ = dx !== nothing ? dx : (length(x_) > 1 ? abs(x_[2] - x_[1]) : one(eltype(x_)))
    dy_ = dy !== nothing ? dy : (length(y_) > 1 ? abs(y_[2] - y_[1]) : one(eltype(y_)))

    F = czt(f, (1.0, 1.0), (1, 2), (neta, nxi))

    dxi  = nx == 1 ? 1.0 : 1.0 / (nxi * dx_)
    deta = ny == 1 ? 1.0 : 1.0 / (neta * dy_)
    xi   = nx == 1 ? [0.0] : collect(range(-(nxi - 1) / 2 * dxi, (nxi - 1) / 2 * dxi, length=nxi))
    eta  = ny == 1 ? [0.0] : collect(range(-(neta - 1) / 2 * deta, (neta - 1) / 2 * deta, length=neta))

    return dx_ * dy_ * F, xi, eta
end

function iczt_ft2(U::AbstractMatrix{<:Number},
                  xi::AbstractVector{<:Real},
                  eta::AbstractVector{<:Real},
                  x_::AbstractVector{<:Real},
                  y_::AbstractVector{<:Real};
                  dx::Union{Real,Nothing}=nothing,
                  dy::Union{Real,Nothing}=nothing)
    nxi  = length(xi)
    neta = length(eta)
    nx_out = length(x_)
    ny_out = length(y_)

    length(x_) == nx_out || throw(ArgumentError("length(x_) must match nx_out"))
    length(y_) == ny_out || throw(ArgumentError("length(y_) must match ny_out"))

    dx_ = dx !== nothing ? dx : (length(x_) > 1 ? abs(x_[2] - x_[1]) : one(eltype(x_)))
    dy_ = dy !== nothing ? dy : (length(y_) > 1 ? abs(y_[2] - y_[1]) : one(eltype(y_)))

    scale_y = ny_out / neta
    scale_x = nx_out / nxi

    u = iczt(U, (scale_y, scale_x), (1, 2), (ny_out, nx_out))

    return u / (dx_ * dy_)
end

end # module

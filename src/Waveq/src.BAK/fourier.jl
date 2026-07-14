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

    phase_x = cis.(-2π .* xi .* transpose(x_))
    phase_y = cis.(-2π .* eta .* transpose(y_))

    # f is (ny, nx), phase_x is (nxi, nx) -> phase_x * transpose(f) is (nxi, ny)
    # wait, U(eta, xi) = sum f(y, x) e^{-i...x} e^{-i...y}
    # U = phase_y * f * transpose(phase_x)
    F = phase_y * f * transpose(phase_x)

    return dx_ * dy_ * F
end

function czt_ft2(f::AbstractMatrix{<:Number},
                 x_::AbstractVector{<:Real},
                 y_::AbstractVector{<:Real},
                 nxi::Int,
                 neta::Int;
                 dxi::Union{Real,Nothing}=nothing,
                 deta::Union{Real,Nothing}=nothing)
    ny, nx = size(f)
    length(x_) == nx || throw(ArgumentError("length(x_) must match size(f,2)"))
    length(y_) == ny || throw(ArgumentError("length(y_) must match size(f,1)"))

    dx_ = length(x_) > 1 ? abs(x_[2] - x_[1]) : one(eltype(x_))
    dy_ = length(y_) > 1 ? abs(y_[2] - y_[1]) : one(eltype(y_))

    dxi_ = dxi !== nothing ? dxi : (nx == 1 ? 1.0 : 1.0 / (nxi * dx_))
    deta_ = deta !== nothing ? deta : (ny == 1 ? 1.0 : 1.0 / (neta * dy_))

    xi   = nxi == 1 ? [0.0] : collect(range(-(nxi - 1) / 2 * dxi_, (nxi - 1) / 2 * dxi_, length=nxi))
    eta  = neta == 1 ? [0.0] : collect(range(-(neta - 1) / 2 * deta_, (neta - 1) / 2 * deta_, length=neta))

    if nx == 1 && ny == 1 && nxi == 1 && neta == 1
        F = f .* (dx_ * dy_)
        return F, xi, eta
    end

    scale_x = nx == 1 ? 1.0 : 1.0 / (nx * dx_ * dxi_)
    scale_y = ny == 1 ? 1.0 : 1.0 / (ny * dy_ * deta_)

    j_dx = nx == 1 ? [0.0] : x_ .- x_[1]
    l_dy = ny == 1 ? [0.0] : y_ .- y_[1]

    phase_in_y = cis.(-2π .* eta[1] .* l_dy)
    phase_in_x = cis.(-2π .* xi[1] .* transpose(j_dx))
    f_tilde = f .* phase_in_y .* phase_in_x

    dims = Int[]
    ny > 1 && push!(dims, 1)
    nx > 1 && push!(dims, 2)

    F = czt(f_tilde, (scale_y, scale_x), dims, (neta, nxi); src_center=(1,1), dst_center=(1,1))

    phase_out_y = cis.(-2π .* eta .* y_[1])
    phase_out_x = cis.(-2π .* transpose(xi) .* x_[1])
    F .*= phase_out_y .* phase_out_x .* (dx_ * dy_)

    return F, xi, eta
end

function iczt_ft2(U::AbstractMatrix{<:Number},
                  xi::AbstractVector{<:Real},
                  eta::AbstractVector{<:Real},
                  x_::AbstractVector{<:Real},
                  y_::AbstractVector{<:Real})
    nxi  = length(xi)
    neta = length(eta)
    nx_out = length(x_)
    ny_out = length(y_)

    length(x_) == nx_out || throw(ArgumentError("length(x_) must match nx_out"))
    length(y_) == ny_out || throw(ArgumentError("length(y_) must match ny_out"))

    dx_ = length(x_) > 1 ? abs(x_[2] - x_[1]) : one(eltype(x_))
    dy_ = length(y_) > 1 ? abs(y_[2] - y_[1]) : one(eltype(y_))

    dxi_ = length(xi) > 1 ? abs(xi[2] - xi[1]) : one(eltype(xi))
    deta_ = length(eta) > 1 ? abs(eta[2] - eta[1]) : one(eltype(eta))

    if nxi == 1 && neta == 1 && nx_out == 1 && ny_out == 1
        u = U .* (dxi_ * deta_)
        return u
    end

    scale_x = nxi == 1 ? 1.0 : 1.0 / (nxi * dx_ * dxi_)
    scale_y = neta == 1 ? 1.0 : 1.0 / (neta * dy_ * deta_)

    p_dxi = nxi == 1 ? [0.0] : xi .- xi[1]
    q_deta = neta == 1 ? [0.0] : eta .- eta[1]

    phase_in_y = cis.(2π .* y_[1] .* q_deta)
    phase_in_x = cis.(2π .* x_[1] .* transpose(p_dxi))
    U_tilde = U .* phase_in_y .* phase_in_x

    dims = Int[]
    neta > 1 && push!(dims, 1)
    nxi > 1 && push!(dims, 2)

    u = iczt(U_tilde, (scale_y, scale_x), dims, (ny_out, nx_out); src_center=(1,1), dst_center=(1,1))

    phase_out_y = cis.(2π .* eta[1] .* y_)
    phase_out_x = cis.(2π .* xi[1] .* transpose(x_))
    
    # iczt divides by N_in. To get sum * dxi, multiply by N_in * dxi.
    factor_x = nxi == 1 ? dxi_ : (nxi * dxi_)
    factor_y = neta == 1 ? deta_ : (neta * deta_)

    u .*= phase_out_y .* phase_out_x .* (factor_x * factor_y)

    return u
end

end # module

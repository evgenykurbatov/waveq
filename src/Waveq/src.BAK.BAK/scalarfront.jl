module scalarfront

import Base: selectdim, isodd
import ..plane: Plane, xwise_iterator, to_plane
import ..fourier: czt_ft2, iczt_ft2

export nyquist_as_res, nyquist_as_res_alt,
       Wavefront, FromSpectrum, FromField, FromIntensity,
       propagate, propagate_parallel

# ------------------------------------------------------------------
# Nyquist sampling
# ------------------------------------------------------------------

function nyquist_as_res(wv::Real,
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real},
                         z::AbstractVector{<:Real},
                         dx_::Real, dy_::Real)
    xi_max  = 1.0 / dx_
    eta_max = 1.0 / dy_

    val = 1.0 / wv^2 - xi_max^2 - eta_max^2
    if val > 0
        grad_kappa_xi  = -2π * xi_max  / sqrt(val)
        grad_kappa_eta = -2π * eta_max / sqrt(val)
    else
        grad_kappa_xi  = 0.0
        grad_kappa_eta = 0.0
    end

    g_xi_pos  = 2π .* x .+ grad_kappa_xi  .* z
    g_xi_neg  = 2π .* x .- grad_kappa_xi  .* z
    g_eta_pos = 2π .* y .+ grad_kappa_eta .* z
    g_eta_neg = 2π .* y .- grad_kappa_eta .* z

    m2_xi  = max.(g_xi_pos.^2,  g_xi_neg.^2)
    m2_eta = max.(g_eta_pos.^2, g_eta_neg.^2)
    m2     = m2_xi .+ m2_eta

    max_grad_phase2 = maximum(m2)
    dxi = π / sqrt(2.0 * max_grad_phase2)

    return (dxi, dxi)
end

function nyquist_as_res_alt(wv::Real,
                             p::AbstractArray{T,N},
                             dx_::Real, dy_::Real) where {T<:Real,N}
    size(p, N) == 3 || throw(ArgumentError("Last dimension must be 3"))
    x = selectdim(p, N, 1)
    y = selectdim(p, N, 2)
    z = selectdim(p, N, 3)
    return nyquist_as_res(wv, vec(x), vec(y), vec(z), dx_, dy_)
end

# ------------------------------------------------------------------
# Wavefront
# ------------------------------------------------------------------

struct FromSpectrum end
struct FromField end
struct FromIntensity end

struct Wavefront{T<:Real}
    pl1::Plane{T}
    U::Matrix{Complex{T}}
    xi::Vector{T}
    eta::Vector{T}
    dxi::T
    deta::T
    u::Union{Matrix{Complex{T}}, Nothing}
end

function _spectrum_size(dx_::Real, d::Real)
    n = round(Int, 1.0 / dx_ / d)
    n = max(0, n)
    2n + 1
end

function Wavefront(::FromSpectrum, pl1::Plane{T},
                   U::AbstractMatrix{<:Complex},
                   dxi::Real, deta::Real) where T<:Real
    neta, nxi = size(U)
    isodd(nxi) || throw(ArgumentError("U must have odd number of columns (nxi=$nxi)"))
    isodd(neta) || throw(ArgumentError("U must have odd number of rows (neta=$neta)"))

    if nxi == 1
        dxi == 1.0 || throw(ArgumentError("Dummy x dimension requires dxi == 1.0"))
        xi = T[0.0]
        dxi_out = T(dxi)
    else
        xi_max = (nxi - 1) / 2 * T(dxi)
        xi = range(-xi_max, xi_max; length=nxi)
        dxi_out = T(dxi)
    end

    if neta == 1
        deta == 1.0 || throw(ArgumentError("Dummy y dimension requires deta == 1.0"))
        eta = T[0.0]
        deta_out = T(deta)
    else
        eta_max = (neta - 1) / 2 * T(deta)
        eta = range(-eta_max, eta_max; length=neta)
        deta_out = T(deta)
    end

    return Wavefront{T}(pl1, Matrix{Complex{T}}(U), collect(xi), collect(eta),
                        dxi_out, deta_out, nothing)
end

function Wavefront(::FromField, pl1::Plane{T},
                   u::AbstractMatrix{<:Complex},
                   dxi::Real, deta::Real) where T<:Real
    ny, nx = size(u)
    pl_nx, pl_ny = pl1.num_nodes
    (ny == pl_ny && nx == pl_nx) ||
        throw(ArgumentError("u must have shape $(pl_ny)×$(pl_nx) matching the plane"))

    dx_ = pl1.dx_
    dy_ = pl1.dy_
    span_x = pl1.span[1]
    span_y = pl1.span[2]

    if nx == 1
        nxi = 1
        dxi_adj = T(dxi)
        x_ = T[0.0]
    else
        nxi = _spectrum_size(dx_, dxi)
        dxi_adj = T(1.0 / (nxi * dx_))
        x_ = collect(range(-span_x / 2, span_x / 2; length=nx))
    end

    if ny == 1
        neta = 1
        deta_adj = T(deta)
        y_ = T[0.0]
    else
        neta = _spectrum_size(dy_, deta)
        deta_adj = T(1.0 / (neta * dy_))
        y_ = collect(range(-span_y / 2, span_y / 2; length=ny))
    end

    U, xi_vec, eta_vec = czt_ft2(u, x_, y_, nxi, neta; dx=pl1.dx_, dy=pl1.dy_)

    return Wavefront{T}(pl1, U, xi_vec, eta_vec, dxi_adj, deta_adj,
                        Matrix{Complex{T}}(u))
end

function Wavefront(::FromIntensity, pl1::Plane{T},
                   im::AbstractMatrix{<:Real},
                   dxi::Real, deta::Real) where T<:Real
    any(im .< 0) && throw(ArgumentError("Intensity must be non-negative"))
    u = Complex{T}.(sqrt.(im))
    return Wavefront(FromField(), pl1, u, dxi, deta)
end

# ------------------------------------------------------------------
# Direct propagation via stripe-based matrix-vector products
# ------------------------------------------------------------------

function _compute_kappa!(K::AbstractMatrix{<:Complex},
                         xi::AbstractVector{<:Real},
                         eta::AbstractVector{<:Real},
                         wv::Real)
    nxi  = length(xi)
    neta = length(eta)
    lam_inv2 = 1.0 / wv^2
    @inbounds for p in 1:nxi
        xi_p = xi[p]
        xi_p2 = xi_p * xi_p
        for q in 1:neta
            rad2 = xi_p2 + eta[q]^2
            if lam_inv2 > rad2
                K[q, p] = 2π * sqrt(lam_inv2 - rad2)
            else
                K[q, p] = 2π * im * sqrt(rad2 - lam_inv2)
            end
        end
    end
    return K
end

"""
    propagate(wf::Wavefront, wv::Real, pl2::Plane; max_mem_mb::Real=2048)

Direct angular-spectrum propagation to an arbitrarily-tilted plane `pl2`.

The integral

    u(x,y) = Δξ Δη Σ_{p,q} e^{i2π(ξ_p x + η_q y)} e^{i κ(ξ_p,η_q) z} U_{pq}

is evaluated stripe-by-stripe over the target plane using BLAS `gemv`.
Each stripe of height `sy` forms a dense matrix `A` of size
`(nx·sy) × (neta·nxi)` and computes `u = A · vec(U)` in one call.

The stripe height is chosen automatically so that `A` fits into the
memory budget `max_mem_mb`.
"""
function propagate(wf::Wavefront{T}, wv::Real, pl2::Plane{T};
                   max_mem_mb::Real=2048) where T<:Real
    nxi  = length(wf.xi)
    neta = length(wf.eta)
    nx2  = pl2.num_nodes[1]
    ny2  = pl2.num_nodes[2]

    # Precompute longitudinal wavenumber matrix κ(ξ,η)
    K = Matrix{Complex{T}}(undef, neta, nxi)
    _compute_kappa!(K, wf.xi, wf.eta, wv)

    # Output buffer
    u_out = Matrix{Complex{T}}(undef, ny2, nx2)

    # Choose stripe height sy so that the phase matrix fits in memory.
    bytes_per_elem = sizeof(Complex{T})
    mem_per_row    = nx2 * neta * nxi * bytes_per_elem
    max_mem        = max_mem_mb * 1024^2
    sy = max(1, min(ny2, div(max_mem, mem_per_row)))

    it = xwise_iterator(pl2, sy)
    y_start = 1

    for (xs, ys, zs) in it
        sy_eff = size(xs, 2)

        # Stack global coordinates -> (nx, sy, 3)
        p_global = similar(xs, nx2, sy_eff, 3)
        p_global[:, :, 1] .= xs
        p_global[:, :, 2] .= ys
        p_global[:, :, 3] .= zs

        # Transform to source-plane local coordinates (x_, y_, z_)
        p_local = to_plane(wf.pl1, p_global)
        x_ = @view p_local[:, :, 1]
        y_ = @view p_local[:, :, 2]
        z_ = @view p_local[:, :, 3]

        # Phase factors for the whole stripe
        #   phase_x[j, p, l] = cis(2π ξ_p x_{jl})
        #   phase_y[j, q, l] = cis(2π η_q y_{jl})
        #   phase_z[j, q, p, l] = cis(κ_{qp} z_{jl})
        phase_x = cis.((2π) .* reshape(x_, (nx2, 1, sy_eff)) .*
                             reshape(wf.xi, (1, nxi, 1)))
        phase_y = cis.((2π) .* reshape(y_, (nx2, 1, sy_eff)) .*
                             reshape(wf.eta, (1, neta, 1)))
        phase_z = cis.(reshape(K, (1, neta, nxi, 1)) .*
                             reshape(z_, (nx2, 1, 1, sy_eff)))

        # A[j, q, p, l] = phase_y[j,q,l] * phase_z[j,q,p,l] * phase_x[j,p,l]
        A = reshape(phase_y, (nx2, neta, 1, sy_eff)) .*
            phase_z .*
            reshape(phase_x, (nx2, 1, nxi, sy_eff))

        # Reshape to ((nx·sy) × (neta·nxi)) and multiply by vec(U)
        A_block = reshape(A, (nx2 * sy_eff, neta * nxi))
        u_block = wf.dxi * wf.deta * (A_block * vec(wf.U))
        u_stripe = reshape(u_block, (nx2, sy_eff))

        # Write stripe into output (transpose because u_stripe is (nx, sy))
        u_out[y_start : y_start + sy_eff - 1, :] .= u_stripe'

        y_start += sy_eff
    end

    return u_out
end

# ------------------------------------------------------------------
# Fast parallel-plane propagation via FFT
# ------------------------------------------------------------------

"""
    propagate_parallel(wf::Wavefront, wv::Real, pl2::Plane)

Fast angular-spectrum propagation to a parallel plane `pl2` using FFT.

This is significantly faster than `propagate` for the special case where
`pl2` is parallel to the source plane (constant propagation distance `z`).
The algorithm pads the spatial field to the target grid size, applies the
propagation phase in the frequency domain via FFT, and transforms back.

For `wf` constructed from a spectrum (`FromSpectrum`) without the original
spatial field, this falls back to the general `propagate` method.
"""
function propagate_parallel(wf::Wavefront{T}, wv::Real, pl2::Plane{T}) where T<:Real
    # --- check parallelism ---
    if !isapprox(wf.pl1.orientation, pl2.orientation; atol=1e-10)
        throw(ArgumentError("propagate_parallel requires parallel planes (same orientation)"))
    end
    if !isapprox(wf.pl1.ort_x, pl2.ort_x; atol=1e-10) ||
       !isapprox(wf.pl1.ort_y, pl2.ort_y; atol=1e-10)
        throw(ArgumentError("propagate_parallel requires parallel planes (same axes)"))
    end

    # --- constant propagation distance and in-plane translation ---
    p_local = to_plane(wf.pl1, pl2.origin)
    x0 = p_local[1]
    y0 = p_local[2]
    z  = p_local[3]

    # --- target coordinates in the source-plane local frame ---
    nx2 = pl2.num_nodes[1]
    ny2 = pl2.num_nodes[2]
    x_target = nx2 > 1 ? collect(range(-pl2.span[1]/2, pl2.span[1]/2; length=nx2)) : [0.0]
    y_target = ny2 > 1 ? collect(range(-pl2.span[2]/2, pl2.span[2]/2; length=ny2)) : [0.0]

    # --- propagation + translation phase applied to stored spectrum ---
    nxi  = length(wf.xi)
    neta = length(wf.eta)
    K = Matrix{Complex{T}}(undef, neta, nxi)
    _compute_kappa!(K, wf.xi, wf.eta, wv)

    phase = K .* z .+ (2π) .* (reshape(wf.xi, (1, nxi)) .* x0 .+ reshape(wf.eta, (neta, 1)) .* y0)
    U_prop = wf.U .* cis.(phase)

    # --- inverse CZT to the target grid ---
    return iczt_ft2(U_prop, wf.xi, wf.eta, x_target, y_target;
                    dx=wf.pl1.dx_, dy=wf.pl1.dy_)
end

end # module

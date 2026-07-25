module scalarfront

import Base: selectdim, isodd
import LinearAlgebra: dot
import ..plane: Plane, xwise_iterator, ywise_iterator, to_plane
import ..fourier: czt_ft2, iczt_ft2

export nyquist_wavenum_res,
       Wavefront, FromSpectrum, FromField, FromIntensity,
       propagate, propagate_parallel

# ------------------------------------------------------------------
# Nyquist sampling
# ------------------------------------------------------------------

function nyquist_wavenum_res(wv::Real,
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

function nyquist_wavenum_res(wv::Real,
                        p::AbstractArray{T,N},
                        dx_::Real, dy_::Real) where {T<:Real,N}
    size(p, N) == 3 || throw(ArgumentError("Last dimension must be 3"))
    x = selectdim(p, N, 1)
    y = selectdim(p, N, 2)
    z = selectdim(p, N, 3)
    return nyquist_wavenum_res(wv, vec(x), vec(y), vec(z), dx_, dy_)
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
        dxi_adj = T(dxi)
        x_ = collect(range(-span_x / 2, span_x / 2; length=nx))
    end

    if ny == 1
        neta = 1
        deta_adj = T(deta)
        y_ = T[0.0]
    else
        neta = _spectrum_size(dy_, deta)
        deta_adj = T(deta)
        y_ = collect(range(-span_y / 2, span_y / 2; length=ny))
    end

    U, xi_vec, eta_vec = czt_ft2(u, x_, y_, nxi, neta; dxi=dxi_adj, deta=deta_adj)

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
                   max_mem_mb::Real=2048, band_limited::Bool=true) where T<:Real
    nxi  = length(wf.xi)
    neta = length(wf.eta)
    nx2  = pl2.num_nodes[1]
    ny2  = pl2.num_nodes[2]

    # Precompute longitudinal wavenumber matrix κ(ξ,η)
    K = Matrix{Complex{T}}(undef, neta, nxi)
    _compute_kappa!(K, wf.xi, wf.eta, wv)

    # Precompute gradients of kappa for Nyquist mask
    grad_k_xi = zeros(T, neta, nxi)
    grad_k_eta = zeros(T, neta, nxi)
    lam_inv2 = 1.0 / wv^2
    for p in 1:nxi, q in 1:neta
        rad2 = wf.xi[p]^2 + wf.eta[q]^2
        if lam_inv2 > rad2
            grad_k_xi[q, p]  = -2π * wf.xi[p]  / sqrt(lam_inv2 - rad2)
            grad_k_eta[q, p] = -2π * wf.eta[q] / sqrt(lam_inv2 - rad2)
        end
    end

    # Output buffer
    u_out = Matrix{Complex{T}}(undef, ny2, nx2)

    # Convert arrays to standard types for fast lookup
    xi_vec = wf.xi
    eta_vec = wf.eta
    U_vec = wf.U
    dxi_deta = wf.dxi * wf.deta
    lam_inv2 = 1.0 / wv^2

    pl1_orig = wf.pl1.origin
    pl1_ort_x = wf.pl1.ort_x
    pl1_ort_y = wf.pl1.ort_y
    pl1_ort_norm = wf.pl1.ort_norm

    # Since K and grad are precomputed, we can use them directly inside the inner loop.
    # To maximize speed and minimize memory, we loop over the target points, and for each point, 
    # we accumulate the sum over the spectrum.

    # Precompute longitudinal wavenumber and gradients to avoid repeated square roots
    kappa_mat = Matrix{Complex{T}}(undef, neta, nxi)
    grad_xi_mat = zeros(T, neta, nxi)
    grad_eta_mat = zeros(T, neta, nxi)

    for p in 1:nxi
        xi_p = xi_vec[p]
        xi_p2 = xi_p^2
        for q in 1:neta
            eta_q = eta_vec[q]
            rad2 = xi_p2 + eta_q^2
            if lam_inv2 > rad2
                kappa_mat[q, p] = 2π * sqrt(lam_inv2 - rad2)
                grad_xi_mat[q, p] = -2π * xi_p / sqrt(lam_inv2 - rad2)
                grad_eta_mat[q, p] = -2π * eta_q / sqrt(lam_inv2 - rad2)
            else
                kappa_mat[q, p] = 2π * im * sqrt(rad2 - lam_inv2)
            end
        end
    end

    # We use a 1D thread parallelization over the total number of target points
    total_pts = nx2 * ny2
    Threads.@threads for idx in 1:total_pts
        i = div(idx - 1, ny2) + 1 # x index
        j = rem(idx - 1, ny2) + 1 # y index

        # global coords
        xl = pl2.num_nodes[1] > 1 ? -pl2.span[1]/2 + (i - 1) * pl2.dx_ : zero(pl2.dx_)
        yl = pl2.num_nodes[2] > 1 ? -pl2.span[2]/2 + (j - 1) * pl2.dy_ : zero(pl2.dy_)

        g = pl2.origin .+ xl .* pl2.ort_x .+ yl .* pl2.ort_y

        # transform to local coords of source plane pl1
        p0 = g .- pl1_orig
        x_loc = dot(p0, pl1_ort_x)
        y_loc = dot(p0, pl1_ort_y)
        z_loc = dot(p0, pl1_ort_norm)

        sum_val = zero(Complex{T})

        @inbounds for p in 1:nxi
            xi_p = xi_vec[p]
            phase_x = 2π * xi_p * x_loc

            for q in 1:neta
                U_pq = U_vec[q, p]
                if U_pq == zero(Complex{T})
                    continue
                end

                eta_q = eta_vec[q]
                kappa = kappa_mat[q, p]

                if band_limited
                    if imag(kappa) == 0.0 # propagating
                        grad_phi_xi = 2π * x_loc + grad_xi_mat[q, p] * z_loc
                        grad_phi_eta = 2π * y_loc + grad_eta_mat[q, p] * z_loc
                        if (grad_phi_xi * wf.dxi)^2 + (grad_phi_eta * wf.deta)^2 > π^2
                            continue
                        end
                    else # evanescent
                        if (2π * x_loc * wf.dxi)^2 + (2π * y_loc * wf.deta)^2 > π^2
                            continue
                        end
                    end
                end

                phase_y = 2π * eta_q * y_loc
                phase_z = kappa * z_loc

                sum_val += U_pq * cis(phase_x + phase_y + phase_z)
            end
        end
        u_out[j, i] = sum_val * dxi_deta
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
function propagate_parallel(wf::Wavefront{T}, wv::Real, pl2::Plane{T}; band_limited::Bool=true) where T<:Real
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

    if band_limited
        lam_inv2 = 1.0 / wv^2
        max_x = nx2 > 1 ? maximum(abs.(x_target)) : 0.0
        max_y = ny2 > 1 ? maximum(abs.(y_target)) : 0.0

        for p in 1:nxi
            xi_p = wf.xi[p]
            for q in 1:neta
                eta_q = wf.eta[q]
                rad2 = xi_p^2 + eta_q^2
                if lam_inv2 > rad2
                    grad_k_xi = -2π * xi_p / sqrt(lam_inv2 - rad2)
                    grad_k_eta = -2π * eta_q / sqrt(lam_inv2 - rad2)

                    # Max magnitude of 2π x + grad_k_xi * z for x in [-max_x, max_x]
                    val_xi_1 = 2π * max_x + grad_k_xi * z
                    val_xi_2 = -2π * max_x + grad_k_xi * z
                    max_grad_phi_xi2 = max(val_xi_1^2, val_xi_2^2)

                    val_eta_1 = 2π * max_y + grad_k_eta * z
                    val_eta_2 = -2π * max_y + grad_k_eta * z
                    max_grad_phi_eta2 = max(val_eta_1^2, val_eta_2^2)

                    if (max_grad_phi_xi2 * wf.dxi^2 + max_grad_phi_eta2 * wf.deta^2) > π^2
                        U_prop[q, p] = 0.0
                    end
                end
            end
        end
    end

    # --- inverse CZT to the target grid ---
    return iczt_ft2(U_prop, wf.xi, wf.eta, x_target, y_target)
end

end # module

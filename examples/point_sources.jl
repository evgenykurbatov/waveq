import Pkg
Pkg.activate(joinpath(@__DIR__, "../src/Waveq"))

# Prevent a window from flickering open during script execution
ENV["GKSwstype"] = "100"

using Plots
using LinearAlgebra
using SpecialFunctions
include("../src/Waveq/src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, nyquist_wavenum_res, propagate_parallel

wv = 0.650 # 650 nm in um
L = 4096 * wv
dx_ = 2.0 * wv
nx = round(Int, L / dx_)
nx = iseven(nx) ? nx + 1 : nx

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, 1.0), num_nodes=(nx, 1))

x_ = range(-L/2, L/2, length=nx)
u0 = zeros(ComplexF64, 1, nx)

idx1 = argmin(abs.(x_ .- (-L/100)))
idx2 = argmin(abs.(x_ .- (L/100)))
u0[1, idx1] = 1.0 / dx_
u0[1, idx2] = 1.0 / dx_

# Nyquist resolutions
z_dist = 1e5 * wv
dxi, deta = nyquist_wavenum_res(wv, [x_[1], x_[end]], [0.0], [z_dist], dx_, 1.0)
dxi *= 6.75
# we use 1D, so deta is not strictly constrained, we can just use 1.0 for it.
wf = Wavefront(FromField(), pl1, u0, dxi, 1.0)

pl2 = Plane([0.0, 0.0, z_dist]; span=(L, 1.0), num_nodes=(nx, 1))

u_exact = zeros(ComplexF64, nx)
k = 2π / wv
for i in 1:nx
    x = x_[i]

    R1 = sqrt((x - x_[idx1])^2 + z_dist^2)
    # 2D Rayleigh-Sommerfeld kernel (line source)
    u1 = (im * k * z_dist / (2 * R1)) * hankelh1(1, k * R1)

    R2 = sqrt((x - x_[idx2])^2 + z_dist^2)
    u2 = (im * k * z_dist / (2 * R2)) * hankelh1(1, k * R2)

    u_exact[i] = u1 + u2
end

u_num_filtered = propagate_parallel(wf, wv, pl2; band_limited=true)
u_num_unfiltered = propagate_parallel(wf, wv, pl2; band_limited=false)

I_exact = abs2.(u_exact)
I_filtered = abs2.(vec(u_num_filtered))
I_unfiltered = abs2.(vec(u_num_unfiltered))

# Subplot 1: Source plane (Intensity)
p1 = plot(x_, I_exact, label="Exact (2D)", c=:gray, alpha=0.75, lw=3, title="Target plane")
plot!(p1, x_, I_unfiltered, label="Unfiltered", c=:cyan, alpha=0.75, ls=:dot, lw=2)
plot!(p1, x_, I_filtered, label="Filtered", c=:red, alpha=0.75, ls=:dash, lw=2)
xlabel!(p1, "x [um]")
ylabel!(p1, "Intensity")

# Calculate spectra for plotting
nxi = length(wf.xi)
K = zeros(ComplexF64, 1, nxi)
lam_inv2 = 1.0 / wv^2
nyquist_mask_x0 = zeros(Bool, nxi)
U2_filtered = copy(wf.U)

max_x = maximum(abs.(x_))
for p in 1:nxi
    xi_p = wf.xi[p]
    rad2 = xi_p^2
    if lam_inv2 > rad2
        K[1, p] = 2π * sqrt(lam_inv2 - rad2)
        grad_k_xi = -2π * xi_p / sqrt(lam_inv2 - rad2)

        # Mask at x=0
        if (grad_k_xi * z_dist * wf.dxi)^2 <= π^2
            nyquist_mask_x0[p] = true
        end

        # Mask at all x (as in propagate_parallel)
        val_xi_1 = 2π * max_x + grad_k_xi * z_dist
        val_xi_2 = -2π * max_x + grad_k_xi * z_dist
        max_grad_phi_xi2 = max(val_xi_1^2, val_xi_2^2)

        if (max_grad_phi_xi2 * wf.dxi^2) > π^2
            U2_filtered[1, p] = 0.0
        end
    else
        K[1, p] = 2π * im * sqrt(rad2 - lam_inv2)
        U2_filtered[1, p] = 0.0
    end
end

U2_unfiltered = wf.U .* cis.(K .* z_dist)
U2_filtered .*= cis.(K .* z_dist)

# Subplot 2: Power spectrum
# using line plots instead of scatter for smaller PDF size
p2 = plot(wf.xi, abs.(vec(wf.U)), c=:gray, alpha=0.5, lw=2, label="Initial", title="Power spectrum")
plot!(p2, wf.xi, abs.(vec(U2_unfiltered)), c=:cyan, alpha=0.5, lw=2, label="Unfiltered")
plot!(p2, wf.xi, abs.(vec(U2_filtered)), c=:red, alpha=0.5, lw=2, label="Filtered")
xlabel!(p2, "ξ [um⁻¹]")
plot!(p2, xscale=:log10, yscale=:log10, ylims=(1e-10, maximum(abs.(wf.U))*10), xlims=(1e-4, maximum(wf.xi)))

valid_xi = wf.xi[nyquist_mask_x0]
if !isempty(valid_xi)
    vline!(p2, [maximum(valid_xi)], c=:black, lw=2, label="")
end

# Subplot 3: Nyquist filter (x=0)
p3 = plot(wf.xi, nyquist_mask_x0, title="Nyquist filter (x=0)", label="", color=:black, lw=2)
xlabel!(p3, "ξ [um⁻¹]")

fig = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400), margin=5Plots.mm)

savefig(fig, joinpath(@__DIR__, "point_sources.pdf"))

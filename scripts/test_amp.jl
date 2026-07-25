using Plots
using LinearAlgebra
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, propagate_parallel

wv = 0.650 # 650 nm in um
L = 2048 * wv
dx_ = 2 * wv
nx = round(Int, L / dx_)
nx = iseven(nx) ? nx + 1 : nx

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, 1.0), num_nodes=(nx, 1))
x_ = range(-L/2, L/2, length=nx)
u0 = zeros(ComplexF64, 1, nx)

idx1 = argmin(abs.(x_ .- (-L/8)))
idx2 = argmin(abs.(x_ .- (L/8)))
u0[1, idx1] = 1.0 / dx_
u0[1, idx2] = 1.0 / dx_

z_dist = 50 * L
dxi, deta = nyquist_as_res(wv, [x_[1], x_[end]], [0.0], [z_dist], dx_, 1.0)
wf = Wavefront(FromField(), pl1, u0, dxi, 1.0)
pl2 = Plane([0.0, 0.0, z_dist]; span=(L, 1.0), num_nodes=(nx, 1))

u_exact = zeros(ComplexF64, nx)
k = 2π / wv
for i in 1:nx
    x = x_[i]
    R1 = sqrt((x - x_[idx1])^2 + z_dist^2)
    u1 = (1.0 / (2π * R1) + k / (im * 2π)) * z_dist / R1^2 * cis(k * R1)
    
    R2 = sqrt((x - x_[idx2])^2 + z_dist^2)
    u2 = (1.0 / (2π * R2) + k / (im * 2π)) * z_dist / R2^2 * cis(k * R2)
    
    u_exact[i] = u1 + u2
end

u_num_filtered = propagate_parallel(wf, wv, pl2; band_limited=true)

println("Exact peak amp: ", maximum(abs.(u_exact)))
println("Num peak amp: ", maximum(abs.(u_num_filtered)))

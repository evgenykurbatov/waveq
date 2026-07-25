import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Prevent a window from flickering open during script execution
ENV["GKSwstype"] = "100"

using Plots
using LinearAlgebra
using SpecialFunctions
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, nyquist_wavenum_res, propagate_parallel, propagate

wv = 0.650 # 650 nm in um
L = 256 * wv
dx_ = 2.0 * wv
dy_ = dx_
nx = round(Int, L / dx_)
nx = iseven(nx) ? nx + 1 : nx
ny = nx

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, L), num_nodes=(nx, ny))

x_ = collect(range(-L/2, L/2, length=nx))
y_ = collect(range(-L/2, L/2, length=ny))
u0 = zeros(ComplexF64, ny, nx)

idx_x1 = argmin(abs.(x_ .- (-L/100)))
idx_x2 = argmin(abs.(x_ .- (L/100)))
idx_y = argmin(abs.(y_ .- 0.0))

# To represent a Dirac delta function of amplitude A=1 on a discrete grid,
# we set the pixel value to A / area.
A = 1.0
u0[idx_y, idx_x1] = A / (dx_ * dy_)
u0[idx_y, idx_x2] = A / (dx_ * dy_)

# Nyquist resolutions
z_dist = 100 * wv
dxi, deta = nyquist_wavenum_res(wv, [x_[1], x_[end]], [y_[1], y_[end]], [z_dist], dx_, dy_)

wf = Wavefront(FromField(), pl1, u0, dxi, deta)
println("nxi: ", length(wf.xi), ", neta: ", length(wf.eta))

pl2 = Plane([0.0, 0.0, z_dist]; span=(L, L), num_nodes=(nx, ny))

# Exact direct sum for 3D propagation (2D target plane)
u_exact = zeros(ComplexF64, ny, nx)
k = 2π / wv
println("Computing exact reference...")
t_exact = @elapsed begin
    for j in 1:ny
        y = y_[j]
        for i in 1:nx
            x = x_[i]

            R1 = sqrt((x - x_[idx_x1])^2 + (y - y_[idx_y])^2 + z_dist^2)
            u1 = (A / (2π)) * (1.0/R1 - im * k) * (z_dist * exp(im * k * R1) / R1^2)

            R2 = sqrt((x - x_[idx_x2])^2 + (y - y_[idx_y])^2 + z_dist^2)
            u2 = (A / (2π)) * (1.0/R2 - im * k) * (z_dist * exp(im * k * R2) / R2^2)

            u_exact[j, i] = u1 + u2
        end
    end
end
println("Exact reference computed in $(round(t_exact, digits=3)) seconds.")

println("Computing filtered SIMD propagation...")
t_filtered = @elapsed u_num_filtered = propagate(wf, wv, pl2; band_limited=true, method=:simd)
println("Filtered SIMD computed in $(round(t_filtered, digits=3)) seconds.")

println("Computing unfiltered SIMD propagation...")
t_unfiltered = @elapsed u_num_unfiltered = propagate(wf, wv, pl2; band_limited=false, method=:simd)
println("Unfiltered SIMD computed in $(round(t_unfiltered, digits=3)) seconds.")

I_exact = abs2.(u_exact[idx_y, :])
I_filtered = abs2.(u_num_filtered[idx_y, :])
I_unfiltered = abs2.(u_num_unfiltered[idx_y, :])

# Plot 1: 1D cross section at y=0
p1 = plot(x_, I_exact, label="Exact (3D)", c=:gray, alpha=0.75, lw=3, title="Target plane (y=0 slice)")
plot!(p1, x_, I_unfiltered, label="Unfiltered", c=:cyan, alpha=0.75, ls=:dot, lw=2)
plot!(p1, x_, I_filtered, label="Filtered", c=:red, alpha=0.75, ls=:dash, lw=2)
xlabel!(p1, "x [um]")
ylabel!(p1, "Intensity")

# Plot 2: 2D Exact Heatmap
p2 = heatmap(x_, y_, abs2.(u_exact), title="Exact Heatmap", colormap=:viridis, aspect_ratio=1)

# Plot 3: 2D Filtered Heatmap
p3 = heatmap(x_, y_, abs2.(u_num_filtered), title="Filtered Heatmap", colormap=:viridis, aspect_ratio=1)

fig = plot(p1, p2, p3, layout=(1, 3), size=(1500, 400), margin=5Plots.mm)

savefig(fig, joinpath(@__DIR__, "point_sources_3d.pdf"))

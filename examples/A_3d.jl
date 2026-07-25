import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

ENV["GKSwstype"] = "100"

using Plots
using LinearAlgebra
using Base.Threads
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromIntensity, nyquist_wavenum_res, propagate_parallel

wv = 0.633 # 633 nm in um
z_dist = 1000 * wv 

nx = 512
ny = 512
# Increased dx and dy so we aren't resolving the evanescent spectrum.
# Using 2 * wv means the maximum spatial frequency is 1/(4*wv), 
# which is well within the propagating regime (1/wv).
dx_ = 2 * wv
dy_ = 2 * wv
L_x = nx * dx_
L_y = ny * dy_

im_gray = zeros(Float64, ny, nx)
# Draw a simple shape (e.g. a square "A")
for j in 1:ny, i in 1:nx
    x = (i - nx/2) * dx_
    y = (j - ny/2) * dy_
    if abs(x) < L_x/4 && abs(y) < L_y/4
        im_gray[j, i] = 1.0
    else
        im_gray[j, i] = 0.1
    end
end

pl1 = Plane([0.0, 0.0, 0.0]; span=(L_x, L_y), num_nodes=(nx, ny))
x_ = [Waveq.plane._x_local(pl1, i) for i in 1:nx]
y_ = [Waveq.plane._y_local(pl1, j) for j in 1:ny]

dxi, deta = nyquist_wavenum_res(wv, [x_[1], x_[end]], [y_[1], y_[end]], [z_dist], dx_, dy_)

wf = Wavefront(FromIntensity(), pl1, im_gray, dxi, deta)

pl2 = Plane([0.0, 0.0, z_dist]; span=(L_x, L_y), num_nodes=(nx, ny))
x_target = [Waveq.plane._x_local(pl2, i) for i in 1:nx]
y_target = [Waveq.plane._y_local(pl2, j) for j in 1:ny]

u_prop = propagate_parallel(wf, wv, pl2; band_limited=true)
I_prop = abs2.(u_prop)

# Calculate and print energy
energy_source = sum(im_gray) * dx_ * dy_
energy_target = sum(I_prop) * dx_ * dy_

println("Energy in Source Plane: ", energy_source)
println("Energy in Target Plane: ", energy_target)
println("Energy Ratio (Target/Source): ", energy_target / energy_source)

# Subplot 1: Source
p1 = heatmap(x_, y_, im_gray, c=:grays, aspect_ratio=1, title="Source", xlabel="x [um]", ylabel="y [um]", yflip=true)

# Subplot 2: Propagation
p2 = heatmap(x_target, y_target, I_prop, c=:grays, aspect_ratio=1, title="Propagation", xlabel="x [um]", ylabel="y [um]", yflip=true)

fig = plot(p1, p2, layout=(1, 2), size=(1200, 600), margin=5Plots.mm)
savefig(fig, joinpath(@__DIR__, "A_3d.pdf"))

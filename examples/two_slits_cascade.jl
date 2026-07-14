import Pkg
Pkg.activate(joinpath(@__DIR__, "../src/Waveq"))
ENV["GKSwstype"] = "100"

using NPZ
using Plots
using LinearAlgebra
using Interpolations
include("../src/Waveq/src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, nyquist_wavenum_res, propagate, propagate_parallel

d = npzread(joinpath(@__DIR__, "two_slits_field.npz"))
wv = d["wv"]
X = d["x"]
Z = d["z"]
u = d["u"]

println("X size: ", length(X))
println("Z size: ", length(Z))
println("u size: ", size(u))

band_limited = false

N = length(X)
L = X[end] - X[1]

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, 1.0), num_nodes=(N, 1))
x1 = collect(range(-L/2, L/2, length=N))
dx1 = x1[2] - x1[1]

z_max = maximum(Z)
dxi, deta = nyquist_wavenum_res(wv, [x1[1], x1[end]], [0.0], [z_max], dx1, 1.0)
dxi *= 2

u1 = reshape(u[1, :], 1, N)
wf1 = Wavefront(FromField(), pl1, u1, dxi, 1.0)

a = [-0.5*L, 40*L]
b = [0.5*L, 40.25*L]
c = 0.5 .* (a .+ b)
d_vec = b .- a
theta = 0.5*π - atan(d_vec[1], d_vec[2])

pl2 = Plane([c[1], 0.0, c[2]]; span=(norm(d_vec), 1.0), orientation=(theta, 0.0), num_nodes=(N, 1))

println("Propagating u1 to pl2...")
@time u2 = propagate(wf1, wv, pl2; band_limited=band_limited)
wf2 = Wavefront(FromField(), pl2, u2, dxi, 1.0)

pl3 = Plane([0.0, 0.0, z_max]; span=(L, 1.0), num_nodes=(N, 1))

println("Propagating u2 to pl3...")
@time u3 = propagate(wf2, wv, pl3; band_limited=band_limited)

itp = interpolate((Z, X), u, Gridded(Constant()))
etp = extrapolate(itp, 0.0)

x2_global = zeros(N)
z2_global = zeros(N)
u2_interp = zeros(ComplexF64, N)
for i in 1:N
    xl = -pl2.span[1]/2 + (i - 1) * pl2.dx_
    g = pl2.origin .+ xl .* pl2.ort_x
    x2_global[i] = g[1]
    z2_global[i] = g[3]
    u2_interp[i] = etp(g[3], g[1])
end

im2_interp = abs2.(u2_interp)
im2 = abs2.(vec(u2))
im3 = abs2.(vec(u3))

println("Interp. Flux 1: ", sum(abs2.(u[1, :])) * dx1)
println("Interp. Flux 2: ", sum(im2_interp) * pl2.dx_)
println("Prop. Flux 2:   ", sum(im2) * pl2.dx_)
println("Interp. Flux 3: ", sum(abs2.(u[end, :])) * dx1)
println("Prop. Flux 3:   ", sum(im3) * pl3.dx_)

p1_plot = plot(x1./L, abs2.(u[end, :]), alpha=0.5, label="Field", title="Final plane")
plot!(p1_plot, x1./L, im3, alpha=0.5, label="Cascade")
xlabel!(p1_plot, "x/L")

x2_local = collect(range(-pl2.span[1]/2, pl2.span[1]/2, length=N))
p2_plot = plot(x2_local./L, im2_interp, alpha=0.5, label="Field", title="Middle plane")
plot!(p2_plot, x2_local./L, im2, alpha=0.5, label="Cascade")
xlabel!(p2_plot, "x_local/L")

p3_plot = heatmap(X./L, Z./L, log10.(abs2.(u) .+ 1e-16), c=:plasma)
scatter!(p3_plot, x2_global./L, z2_global./L, c=:black, m=:circle, ms=1, label="")
scatter!(p3_plot, x1./L, fill(z_max/L, N), c=:black, m=:circle, ms=1, label="")
xlabel!(p3_plot, "x/L")
ylabel!(p3_plot, "z/L")

l = @layout [grid(2, 1) a{0.5w}]
fig = plot(p1_plot, p2_plot, p3_plot, layout=l, size=(1000, 800))

savefig(fig, joinpath(@__DIR__, "two_slits_cascade.pdf"))
println("Saved two_slits_cascade.pdf")

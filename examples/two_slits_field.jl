import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "100"

using Plots
using LinearAlgebra
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromIntensity, nyquist_wavenum_res, propagate_parallel

wv = 0.600 # 600 nm in um
N = 1000
dx_ = 2 * wv
L = dx_ * (N - 1)

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, 1.0), num_nodes=(N, 1))
x1 = collect(range(-L/2, L/2, length=N))

D_end = 100 * L
D = range(0.0, D_end, length=2001)

dxi, deta = nyquist_wavenum_res(wv, [x1[1], x1[end]], [0.0], [D_end], dx_, 1.0)
dxi *= 5

slit_position = 150 * wv
slit_width = 20 * wv

# Gaussian slits
im1 = exp.(-(x1 .+ slit_position).^2 ./ (2 * slit_width^2)) .+
      exp.(-(x1 .- slit_position).^2 ./ (2 * slit_width^2))
im1 = reshape(im1, 1, N)

wf = Wavefront(FromIntensity(), pl1, im1, dxi, 1.0)
println("Wavenumber space M: ", length(wf.xi))

# Simulate the volume
u = zeros(ComplexF64, length(D), N)
println("Propagating steps...")
@time for (i, z) in enumerate(D)
    pl2 = Plane([0.0, 0.0, z]; span=(L, 1.0), num_nodes=(N, 1))
    u[i, :] .= vec(propagate_parallel(wf, wv, pl2; band_limited=false))
end

im = abs2.(u)

im_log = log10.(im .+ 1e-16)

titl_str = string("wv = ", wv*1000, " [nm], L = ", round(L, digits=2), " [um]")
p = heatmap(x1 ./ L, collect(D) ./ L, im_log, 
            c=:plasma, 
            xlabel="x/L", 
            ylabel="z/L", 
            title=titl_str,
            size=(800, 1000), 
            dpi=300)

savefig(p, joinpath(@__DIR__, "two_slits_field.pdf"))
println("Saved two_slits_field.pdf")

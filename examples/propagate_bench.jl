import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BenchmarkTools
using LinearAlgebra
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, propagate, nyquist_wavenum_res

wv = 0.650
L = 100 * wv
N = 100
dx_ = L / N

pl1 = Plane([0.0, 0.0, 0.0]; span=(L, L), num_nodes=(N, N))

u0 = randn(ComplexF64, N, N)

z_dist = 50 * L
theta = deg2rad(10.0)
phi = deg2rad(15.0)
pl2 = Plane([0.0, 0.0, z_dist]; span=(L, L), orientation=(theta, phi), num_nodes=(N, N))

dxi, deta = nyquist_wavenum_res(wv, [-L/2, L/2], [-L/2, L/2], [z_dist], dx_, dx_)
#dxi *= 2.0
#deta *= 2.0

wf = Wavefront(FromField(), pl1, u0, dxi, deta)

total_steps = N^2 * length(wf.xi) * length(wf.eta)

println(string("Target grid: ", N, "x", N))
println(string("Spectrum grid: ", length(wf.xi), "x", length(wf.eta)))
println(string("Total evaluation steps per method: ", total_steps))

u_loop = propagate(wf, wv, pl2; band_limited=true, method=:loop)
u_simd = propagate(wf, wv, pl2; band_limited=true, method=:simd)

println(string("Max Difference (|u_loop - u_simd|): ", maximum(abs.(u_loop .- u_simd))))

println("\nBenchmarking :loop method...")
bench_loop = @benchmark propagate($wf, $wv, $pl2; band_limited=true, method=:loop) samples=5
display(bench_loop)

println("\n\nBenchmarking :simd method...")
bench_simd = @benchmark propagate($wf, $wv, $pl2; band_limited=true, method=:simd) samples=5
display(bench_simd)

ratio = median(bench_loop).time / median(bench_simd).time
println(string("\n\nSIMD speedup ratio: ", round(ratio, digits=2), "x"))

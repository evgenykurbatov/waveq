#!/usr/bin/env julia

# Example: Using Waveq directly from Julia with multithreading
#
# Run with: julia --project=.. --threads=auto scripts/example.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Waveq

println("Julia version: ", VERSION)
println("Threads available: ", Threads.nthreads())
println()

# Build two planes
pl1 = Plane(
    origin=[0.0, 0.0, 0.0],
    span=[2.0, 1.5],
    num_nodes=[4, 3],
    orientation=[0.0, 0.0],
    dtype=Float64,
)

pl2 = Plane(
    origin=[0.0, 0.0, 1.0],
    span=[2.0, 1.5],
    num_nodes=[4, 3],
    orientation=[deg2rad(15.0), deg2rad(45.0)],
    dtype=Float64,
)

println("pl1 origin: ", pl1.origin)
println("pl2 origin: ", pl2.origin)
println("pl1 p shape: ", size(pl1.p))
println("pl2 p shape: ", size(pl2.p))

# Test Nyquist sampling
x = [0.0, 0.5, -0.5, 1.0, -1.0]
y = [0.0, 0.3, -0.3, 0.8, -0.8]
z = [0.05, 0.05, 0.05, 0.05, 0.05]
dxi, deta = nyquist_as_res(632.8e-9, x, y, z, 10e-6, 10e-6)
println("Nyquist dxi = ", dxi, ", deta = ", deta)

# Test CZT
using FFTW
f = rand(ComplexF64, 64, 64)
x_ = range(-1.0, 1.0, length=64)
y_ = range(-1.0, 1.0, length=64)
F, xi, eta = czt_ft2(f, collect(x_), collect(y_), 51, 51)
println("CZT output shape: ", size(F))
println("xi range: ", xi[1], " to ", xi[end])
println("eta range: ", eta[1], " to ", eta[end])

println("\nAll OK")

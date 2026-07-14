#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "src", "Waveq"))

using Waveq
using LinearAlgebra

println("Testing propagate() implementation")
println("="^50)

# ------------------------------------------------------------------
# Test 1: Delta spectrum -> plane wave (exact test of inverse formula)
# ------------------------------------------------------------------
println("\n--- Test 1: Delta spectrum -> plane wave ---")
pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 2.0], num_nodes=[15, 15], dtype=Float64)
dxi, deta = 0.15, 0.15
U_delta = zeros(ComplexF64, 15, 15)
U_delta[8, 8] = 1.0 / (dxi * deta)

wf_delta = Wavefront(FromSpectrum(), pl1, U_delta, dxi, deta)
u_prop = propagate(wf_delta, 632.8e-9, pl1)

rel_var = (maximum(abs.(u_prop)) - minimum(abs.(u_prop))) / maximum(abs.(u_prop))
println("Relative variation (should be ~0): ", rel_var)
@assert rel_var < 1e-10 "Delta spectrum did not produce uniform plane wave"

# ------------------------------------------------------------------
# Test 1b: Gaussian spectrum -> Gaussian field (same plane)
# ------------------------------------------------------------------
println("\n--- Test 1b: Gaussian spectrum reconstruction ---")
# A Gaussian in x,y space has a Gaussian spectrum.
# Construct the spectrum directly and reconstruct.
xi_vec = range(-1.0, 1.0, length=15)
eta_vec = range(-1.0, 1.0, length=15)
U_gauss = ComplexF64[exp(-π*(ξ^2 + η^2)) for η in eta_vec, ξ in xi_vec]
wf_gauss = Wavefront(FromSpectrum(), pl1, U_gauss, dxi, deta)
u_gauss = propagate(wf_gauss, 632.8e-9, pl1)
println("Reconstructed field shape: ", size(u_gauss))
@assert all(isfinite, u_gauss) "Non-finite values in Gaussian reconstruction"
@assert size(u_gauss) == (15, 15) "Wrong output shape"

# ------------------------------------------------------------------
# Test 2: Propagation to a parallel plane at finite distance
# ------------------------------------------------------------------
println("\n--- Test 2: Parallel plane propagation ---")
pl2 = Plane([0.0, 0.0, 0.5]; span=[2.0, 2.0], num_nodes=[15, 15], dtype=Float64)
u_prop2 = propagate(wf_delta, 632.8e-9, pl2)
println("Output shape: ", size(u_prop2))
@assert size(u_prop2) == (15, 15) "Wrong output shape"
@assert all(isfinite, u_prop2) "Non-finite values in output"

# ------------------------------------------------------------------
# Test 3: Propagation to a tilted plane
# ------------------------------------------------------------------
println("\n--- Test 3: Tilted plane propagation ---")
pl3 = Plane([0.0, 0.0, 0.5]; span=[2.0, 2.0], num_nodes=[15, 15],
             orientation=[deg2rad(10.0), deg2rad(20.0)], dtype=Float64)
u_prop3 = propagate(wf_delta, 632.8e-9, pl3)
println("Output shape: ", size(u_prop3))
@assert size(u_prop3) == (15, 15) "Wrong output shape"
@assert all(isfinite, u_prop3) "Non-finite values in output"

# ------------------------------------------------------------------
# Test 4: 2D case (dummy y dimension)
# ------------------------------------------------------------------
println("\n--- Test 4: 2D dummy-y propagation ---")
pl_2d = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[7, 1], dtype=Float64)
u_2d = rand(ComplexF64, 1, 7)
wf_2d = Wavefront(FromField(), pl_2d, u_2d, 0.1, 1.0)

pl_2d_target = Plane([0.0, 0.0, 0.1]; span=[2.0, 1.5], num_nodes=[7, 1], dtype=Float64)
u_2d_prop = propagate(wf_2d, 632.8e-9, pl_2d_target)
println("Output shape: ", size(u_2d_prop))
@assert size(u_2d_prop) == (1, 7) "Wrong output shape for 2D"
@assert all(isfinite, u_2d_prop) "Non-finite values in 2D output"

# ------------------------------------------------------------------
# Test 5: Different target grid size
# ------------------------------------------------------------------
println("\n--- Test 5: Different target grid size ---")
pl4 = Plane([0.0, 0.0, 0.3]; span=[1.5, 1.2], num_nodes=[7, 9], dtype=Float64)
u_prop4 = propagate(wf_delta, 632.8e-9, pl4)
println("Output shape: ", size(u_prop4))
@assert size(u_prop4) == (9, 7) "Wrong output shape for different grid"
@assert all(isfinite, u_prop4) "Non-finite values in output"

# ------------------------------------------------------------------
# Test 6: Memory budget / stripe size
# ------------------------------------------------------------------
println("\n--- Test 6: Stripe size with memory budget ---")
pl_big = Plane([0.0, 0.0, 0.0]; span=[2.0, 2.0], num_nodes=[31, 31], dtype=Float64)
U_big = zeros(ComplexF64, 31, 31)
U_big[16, 16] = 1.0 / (0.1 * 0.1)
wf_big = Wavefront(FromSpectrum(), pl_big, U_big, 0.1, 0.1)

# Use small memory budget to force multiple stripes
u_big = propagate(wf_big, 632.8e-9, pl_big; max_mem_mb=1)
println("Output shape: ", size(u_big))
@assert size(u_big) == (31, 31) "Wrong output shape"
@assert all(isfinite, u_big) "Non-finite values in output"
rel_var_big = (maximum(abs.(u_big)) - minimum(abs.(u_big))) / maximum(abs.(u_big))
println("Relative variation (should be ~0): ", rel_var_big)
@assert rel_var_big < 1e-10 "Multi-stripe propagation corrupted plane wave"

println("\n" * "="^50)
println("All tests passed!")

using Waveq

# --- 2D case (ny=1, dummy y) ---
pl_2d = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[7, 1], orientation=[0.0, 0.0], dtype=Float64)

u_2d = rand(ComplexF64, 1, 7)  # ny=1, nx=7
wf_2d = Wavefront(FromField(), pl_2d, u_2d, 0.1, 1.0)
println("2D FromField: U shape = ", size(wf_2d.U), ", xi len = ", length(wf_2d.xi), ", eta len = ", length(wf_2d.eta))
println("  dxi = ", wf_2d.dxi, ", deta = ", wf_2d.deta)
println("  xi[1] = ", wf_2d.xi[1], ", xi[end] = ", wf_2d.xi[end])
println("  eta = ", wf_2d.eta)
println("  xi odd? ", isodd(length(wf_2d.xi)))

# --- 3D case ---
pl_3d = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[7, 5], orientation=[0.0, 0.0], dtype=Float64)

# From spectrum U with free dxi/deta
U = rand(ComplexF64, 9, 11)
wf1 = Wavefront(FromSpectrum(), pl_3d, U, 0.2, 0.5)
println("\nFromSpectrum: U shape = ", size(wf1.U), ", xi len = ", length(wf1.xi), ", eta len = ", length(wf1.eta))
println("  dxi = ", wf1.dxi, ", deta = ", wf1.deta)
println("  xi[1] = ", wf1.xi[1], ", xi[end] = ", wf1.xi[end])
println("  eta[1] = ", wf1.eta[1], ", eta[end] = ", wf1.eta[end])
println("  xi odd? ", isodd(length(wf1.xi)), ", eta odd? ", isodd(length(wf1.eta)))

# From field u (matches plane shape: ny=5, nx=7)
u = rand(ComplexF64, 5, 7)
wf2 = Wavefront(FromField(), pl_3d, u, 0.1, 0.1)
println("\nFromField: U shape = ", size(wf2.U), ", xi len = ", length(wf2.xi), ", eta len = ", length(wf2.eta))
println("  u stored? ", wf2.u !== nothing)
println("  dxi = ", wf2.dxi, ", deta = ", wf2.deta)
println("  xi odd? ", isodd(length(wf2.xi)), ", eta odd? ", isodd(length(wf2.eta)))

# From intensity im (real, matches plane shape)
im = rand(Float64, 5, 7)
wf3 = Wavefront(FromIntensity(), pl_3d, im, 0.1, 0.1)
println("\nFromIntensity: U shape = ", size(wf3.U), ", xi len = ", length(wf3.xi), ", eta len = ", length(wf3.eta))
println("  u stored? ", wf3.u !== nothing)

# Dummy propagate
try
    propagate(wf1, 632.8e-9, pl_3d)
catch e
    println("\nPropagate dummy: ", e)
end

println("\nOK")

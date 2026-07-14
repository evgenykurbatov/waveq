using Waveq
using LinearAlgebra: norm

function test_same_plane_integer_ratio()
    println("=== Test 1: Same plane (z=0), integer-ratio spectrum ===")
    nx, ny = 63, 47
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    x_ = collect(range(-pl1.span[1]/2, pl1.span[1]/2; length=nx))
    y_ = collect(range(-pl1.span[2]/2, pl1.span[2]/2; length=ny))
    U, xi, eta = czt_ft2(u, x_, y_, 3*nx, 3*ny)
    wf = Wavefront(FromSpectrum(), pl1, U, xi[2]-xi[1], eta[2]-eta[1])
    pl2 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u_prop = propagate_parallel(wf, 1.0, pl2)
    err = norm(u_prop - u) / norm(u)
    println("Relative error vs original field: $err")
    @assert err < 1e-10
    println("PASS\n")
end

function test_same_plane_fromfield()
    println("=== Test 2: Same plane (z=0), FromField wavefront ===")
    nx, ny = 65, 49
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    wf = Wavefront(FromField(), pl1, u, 0.1, 0.1)
    pl2 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u_prop = propagate_parallel(wf, 1.0, pl2)
    err = norm(u_prop - u) / norm(u)
    println("Relative error vs original field: $err")
    @assert err < 0.2
    println("PASS\n")
end

function test_small_z_evolution()
    println("=== Test 3: Small z, field evolution ===")
    nx, ny = 63, 47
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    x_ = collect(range(-pl1.span[1]/2, pl1.span[1]/2; length=nx))
    y_ = collect(range(-pl1.span[2]/2, pl1.span[2]/2; length=ny))
    U, xi, eta = czt_ft2(u, x_, y_, 3*nx, 3*ny)
    wf = Wavefront(FromSpectrum(), pl1, U, xi[2]-xi[1], eta[2]-eta[1])
    pl2 = Plane([0.0, 0.0, 0.1]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u_prop = propagate_parallel(wf, 0.01, pl2)
    
    err_vs_orig = norm(u_prop - u) / norm(u)
    println("Relative change from original: $err_vs_orig")
    @assert err_vs_orig > 0.01 "Field should have evolved after propagation"
    println("PASS\n")
end

function test_translation_evolution()
    println("=== Test 4: In-plane translation ===")
    nx, ny = 63, 47
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    x_ = collect(range(-pl1.span[1]/2, pl1.span[1]/2; length=nx))
    y_ = collect(range(-pl1.span[2]/2, pl1.span[2]/2; length=ny))
    U, xi, eta = czt_ft2(u, x_, y_, 3*nx, 3*ny)
    wf = Wavefront(FromSpectrum(), pl1, U, xi[2]-xi[1], eta[2]-eta[1])
    pl2 = Plane([0.5, 0.25, 0.1]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u_prop = propagate_parallel(wf, 0.01, pl2)
    
    err_vs_orig = norm(u_prop - u) / norm(u)
    println("Relative change from original: $err_vs_orig")
    @assert err_vs_orig > 0.01 "Field should have evolved after propagation"
    println("PASS\n")
end

function test_larger_grid()
    println("=== Test 5: Larger target grid ===")
    nx, ny = 63, 47
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    x_ = collect(range(-pl1.span[1]/2, pl1.span[1]/2; length=nx))
    y_ = collect(range(-pl1.span[2]/2, pl1.span[2]/2; length=ny))
    U, xi, eta = czt_ft2(u, x_, y_, 3*nx, 3*ny)
    wf = Wavefront(FromSpectrum(), pl1, U, xi[2]-xi[1], eta[2]-eta[1])
    pl2 = Plane([0.0, 0.0, 0.1]; span=[2.0, 1.5], num_nodes=[129, 97], dtype=Float64)
    u_prop = propagate_parallel(wf, 0.01, pl2)
    @assert size(u_prop) == (97, 129) "Output size should be 97x129"
    println("Output size: $(size(u_prop))")
    println("PASS\n")
end

function test_non_parallel_plane()
    println("=== Test 6: Non-parallel plane (should error) ===")
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 2.0], num_nodes=[63, 47], dtype=Float64)
    u = rand(ComplexF64, 47, 63)
    wf = Wavefront(FromField(), pl1, u, 0.1, 0.1)
    pl2 = Plane([0.0, 0.0, 0.5]; span=[2.0, 2.0], num_nodes=[63, 47], orientation=[deg2rad(10.0), 0.0], dtype=Float64)
    try
        propagate_parallel(wf, 1.0, pl2)
        println("FAIL: Should have thrown error")
    catch
        println("PASS\n")
    end
end

function test_from_spectrum()
    println("=== Test 7: FromSpectrum wavefront at z=0 ===")
    nx, ny = 63, 47
    pl1 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u = rand(ComplexF64, ny, nx)
    wf_field = Wavefront(FromField(), pl1, u, 0.1, 0.1)
    wf_spec = Wavefront(FromSpectrum(), pl1, wf_field.U, wf_field.dxi, wf_field.deta)
    pl2 = Plane([0.0, 0.0, 0.0]; span=[2.0, 1.5], num_nodes=[nx, ny], dtype=Float64)
    u_prop = propagate_parallel(wf_spec, 1.0, pl2)
    err = norm(u_prop - u) / norm(u)
    println("Relative error vs original field: $err")
    @assert err < 0.2
    println("PASS\n")
end

test_same_plane_integer_ratio()
test_same_plane_fromfield()
test_small_z_evolution()
test_translation_evolution()
test_larger_grid()
test_non_parallel_plane()
test_from_spectrum()
println("All propagate_parallel tests passed!")

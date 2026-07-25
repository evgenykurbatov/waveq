using Test
using LinearAlgebra
include("../src/Waveq.jl")
using .Waveq
using .Waveq: Plane, Wavefront, FromField, propagate, propagate_parallel

@testset "Propagation" begin
    wv = 0.650
    nx = 64
    ny = 64
    pl1 = Plane([0.0, 0.0, 0.0]; span=(20.0, 20.0), num_nodes=(nx, ny))
    
    u0 = randn(ComplexF64, ny, nx)
    wf = Wavefront(FromField(), pl1, u0, 0.1, 0.1)
    
    pl2 = Plane([0.0, 0.0, 10.0]; span=(20.0, 20.0), num_nodes=(nx, ny))
    
    u_par = propagate_parallel(wf, wv, pl2; band_limited=true)
    u_gen = propagate(wf, wv, pl2; band_limited=true)
    
    # They shouldn't be exactly the same because propagate uses discrete sum approximation
    # over x_ and y_, while propagate_parallel uses FFT on the whole spectrum.
    # But they should be roughly the same.
    @test size(u_par) == (ny, nx)
    @test size(u_gen) == (ny, nx)
    
    # test dummy dimensions
    pl1_1d = Plane([0.0, 0.0, 0.0]; span=(20.0, 1.0), num_nodes=(nx, 1))
    u0_1d = randn(ComplexF64, 1, nx)
    wf_1d = Wavefront(FromField(), pl1_1d, u0_1d, 0.1, 1.0)
    pl2_1d = Plane([0.0, 0.0, 10.0]; span=(20.0, 1.0), num_nodes=(nx, 1))
    u_par_1d = propagate_parallel(wf_1d, wv, pl2_1d; band_limited=false)
    u_gen_1d = propagate(wf_1d, wv, pl2_1d; band_limited=false)
    @test size(u_par_1d) == (1, nx)
    @test size(u_gen_1d) == (1, nx)
end

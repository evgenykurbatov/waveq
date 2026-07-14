using Test
using LinearAlgebra
include("../src/Waveq/src/Waveq.jl")
using .Waveq
using .Waveq: nyquist_wavenum_res

@testset "Nyquist Functions" begin
    wv = 0.65
    x = range(-10.0, 10.0, length=10)
    y = range(-5.0, 5.0, length=5)
    z = fill(100.0, 50)
    
    # We test that array input gives same as vector
    p = zeros(50, 3)
    idx = 1
    for xx in x, yy in y
        p[idx, 1] = xx
        p[idx, 2] = yy
        p[idx, 3] = 100.0
        idx += 1
    end
    
    dx_ = 2.0
    dy_ = 2.0
    
    dxi1, deta1 = nyquist_wavenum_res(wv, p[:, 1], p[:, 2], p[:, 3], dx_, dy_)
    dxi2, deta2 = nyquist_wavenum_res(wv, p, dx_, dy_)
    
    @test dxi1 ≈ dxi2
    @test deta1 ≈ deta2
    @test dxi1 > 0
end

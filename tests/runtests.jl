using Test

@testset "Waveq Tests" begin
    include("test_nyquist.jl")
    include("test_propagate.jl")
end

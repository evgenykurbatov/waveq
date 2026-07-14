import Pkg
Pkg.activate(joinpath(@__DIR__, "../src/Waveq"))

# Prevent window
ENV["GKSwstype"] = "100"

using Plots
include("../src/Waveq/src/Waveq.jl")
using .Waveq: SF5, F2, N_BK7, wv_ref_d, refractive_index

wv = range(0.365, 2.5, length=100) # um
wv_nm = wv .* 1e3

p = plot(xlabel="Wavelength [nm]", ylabel="Refractive index", title="Dispersion law", size=(600, 400), legend=:topright, framestyle=:box)

for glass in [SF5, F2, N_BK7]
    n = [refractive_index(glass, w) for w in wv]
    plot!(p, wv_nm, n, label=glass.name)
end

ref_nm = round(Int, wv_ref_d * 1e3)
vline!(p, [wv_ref_d * 1e3], ls=:dot, c=:grey, label=string(ref_nm, " nm"))

savefig(p, joinpath(@__DIR__, "glass_dispersion.pdf"))
println("Saved glass_dispersion.pdf")

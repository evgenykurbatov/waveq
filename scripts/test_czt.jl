using Waveq

nx, ny = 64, 64
Lx, Ly = 2.0, 2.0
x_ = range(-Lx/2, Lx/2, length=nx)
y_ = range(-Ly/2, Ly/2, length=ny)
X = [x for x in x_, y in y_]
Y = [y for x in x_, y in y_]
f = exp.(-π*(X.^2 .+ Y.^2))

nxi, neta = 51, 51
F, xi, eta = czt_ft2(f, collect(x_), collect(y_), nxi, neta)
println("czt_ft2 shape: ", size(F))
println("czt_ft2 xi range: ", xi[1], " to ", xi[end])
println("czt_ft2 eta range: ", eta[1], " to ", eta[end])
println("czt_ft2 center: ", abs(F[neta÷2+1, nxi÷2+1]))

F_direct = direct_ft2(f, collect(x_), collect(y_), xi, eta)
println("direct_ft2 shape: ", size(F_direct))
println("Match: ", isapprox(F, F_direct, rtol=1e-10))

println("OK")

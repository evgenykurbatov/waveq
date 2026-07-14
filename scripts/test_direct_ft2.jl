using Waveq

# Test 1: Gaussian -> should also be Gaussian in frequency domain
nx, ny = 64, 64
Lx, Ly = 2.0, 2.0
x_ = range(-Lx/2, Lx/2, length=nx)
y_ = range(-Ly/2, Ly/2, length=ny)
X = [x for x in x_, y in y_]
Y = [y for x in x_, y in y_]
f = exp.(-π*(X.^2 .+ Y.^2))

# Output grid: odd, symmetric, different size
nxi, neta = 51, 51
xi = range(-5.0, 5.0, length=nxi)
eta = range(-5.0, 5.0, length=neta)

F = direct_ft2(f, collect(x_), collect(y_), collect(xi), collect(eta))
println("Shape: ", size(F))
println("Peak at center: ", abs(F[neta÷2+1, nxi÷2+1]))
println("Center should be near 1.0: ", abs(F[neta÷2+1, nxi÷2+1]) ≈ 1.0)

# Test 2: Check xi=0 symmetry for real input
println("Real input -> conjugate symmetric: ", F[1,1] ≈ conj(F[end,end]))

println("OK")

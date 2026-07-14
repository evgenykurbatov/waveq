using FourierTools

nx = 11
f = randn(ComplexF64, nx)
x_ = collect(range(-5.0, 5.0, length=nx))
dx = x_[2] - x_[1]

nxi = 15
dxi = 0.3
xi = collect(range(-(nxi-1)/2 * dxi, (nxi-1)/2 * dxi, length=nxi))

phase_x = cis.(-2π .* xi .* x_')
F_dir = f' * phase_x' |> vec

scale_x = 1.0 / (nx * dx * dxi)
F_czt = czt(f, scale_x, 1, nxi)

ratio = F_dir ./ F_czt
println("ratio = ", ratio[1:3])
println("diff ratio = ", ratio[2] / ratio[1], " ", ratio[3] / ratio[2])

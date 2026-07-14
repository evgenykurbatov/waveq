using FourierTools
nx = 11
f = randn(ComplexF64, nx)
x_ = collect(range(-5.0, 5.0, length=nx))
dx = x_[2] - x_[1]

nxi = 15
dxi = 0.3
xi = collect(range(-(nxi-1)/2 * dxi, (nxi-1)/2 * dxi, length=nxi))

phase_x = cis.(-2π .* xi .* transpose(x_))
F_dir = transpose(f) * transpose(phase_x) |> vec

scale_x = 1.0 / (nx * dx * dxi)

j_dx = x_ .- x_[1]
f_tilde = f .* cis.(-2π .* xi[1] .* j_dx)

F_czt = czt(f_tilde, scale_x, 1, nxi; src_center=(1,), dst_center=(1,))
F_czt .*= cis.(-2π .* xi .* x_[1])

println("Diff: ", maximum(abs.(F_dir .- F_czt)))

using FourierTools
nxi = 15
U = randn(ComplexF64, nxi)
dxi = 0.3
xi = collect(range(-(nxi-1)/2 * dxi, (nxi-1)/2 * dxi, length=nxi))

nx = 11
x_ = collect(range(-5.0, 5.0, length=nx))
dx = x_[2] - x_[1]

phase_x = cis.(2π .* x_ .* transpose(xi))
u_dir = dxi .* (phase_x * U)

scale_x = 1.0 / (nxi * dx * dxi)

p_dxi = xi .- xi[1]
U_tilde = U .* cis.(2π .* x_[1] .* p_dxi)

u_czt = iczt(U_tilde, scale_x, 1, nx; src_center=(1,), dst_center=(1,))

# test different norm:
ratio = u_dir ./ (u_czt .* cis.(2π .* xi[1] .* x_))
println("ratio = ", ratio[1:3])

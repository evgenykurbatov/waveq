using Waveq
using Plots
using LinearAlgebra

# ------------------------------------------------------------------
# Point-source interferometry test
# ------------------------------------------------------------------

const A = 1.0
const wv = 600e-9
const k = 2π / wv
const source_sep = 300 * wv
const prop_dist = 2000 * wv

# Point source positions: symmetric about origin, behind the source plane
const source1_pos = [-source_sep/2, 0.0, -prop_dist]
const source2_pos = [ source_sep/2, 0.0, -prop_dist]

function point_source_field(x, y, z, x0, y0, z0)
    R = sqrt((x - x0)^2 + (y - y0)^2 + (z - z0)^2)
    return A / (2π) * (1/R - im*k) * (z - z0) * exp(im*k*R) / R^2
end

function exact_field(xs, ys, zs)
    """Compute exact field from two point sources at given coordinates."""
    ny = length(ys)
    nx = length(xs)
    u = zeros(ComplexF64, ny, nx)
    for j in 1:nx, i in 1:ny
        u[i, j] = (point_source_field(xs[j], ys[i], zs, source1_pos...) +
                   point_source_field(xs[j], ys[i], zs, source2_pos...))
    end
    return u
end

function estimate_dxi_deta(pl_target::Plane, wv::Real, dx_::Real, dy_::Real)
    """Use nyquist_as_res with the target plane's corner points."""
    sx, sy = pl_target.span
    corners_local = [
        [ sx/2,  sy/2],
        [ sx/2, -sy/2],
        [-sx/2,  sy/2],
        [-sx/2, -sy/2]
    ]
    corners_global = [from_plane2d(pl_target, c) for c in corners_local]
    x = [c[1] for c in corners_global]
    y = [c[2] for c in corners_global]
    z = [c[3] for c in corners_global]
    return nyquist_as_res(wv, x, y, z, dx_, dy_)
end

# ------------------------------------------------------------------
# Case 1: Flat propagation (ny = 1) to parallel plane
# ------------------------------------------------------------------

function case1_flat_parallel()
    println("=== Case 1: Flat (ny=1) to parallel plane ===")

    span_x = 200 * wv
    nx = 401
    dx_ = span_x / (nx - 1)

    pl_source = Plane([0.0, 0.0, 0.0];
                      span=[span_x, wv],
                      num_nodes=[nx, 1],
                      dtype=Float64)

    xs = collect(range(-span_x/2, span_x/2; length=nx))
    u_source = exact_field(xs, [0.0], 0.0)

    pl_target = Plane([0.0, 0.0, prop_dist];
                      span=[span_x, wv],
                      num_nodes=[nx, 1],
                      dtype=Float64)

    dxi, _ = estimate_dxi_deta(pl_target, wv, dx_, dx_)
    println("  Estimated dxi = $dxi")
    wf = Wavefront(FromField(), pl_source, u_source, dxi, 1.0)
    println("  Actual dxi = $(wf.dxi), nxi = $(length(wf.xi))")

    u_num = propagate_parallel(wf, wv, pl_target)
    u_exact_target = exact_field(xs, [0.0], prop_dist)

    I_num = abs2.(u_num)
    I_exact = abs2.(u_exact_target)

    p = plot(xs / wv, vec(I_exact), label="Exact", linewidth=2, color=:blue,
             xlabel="x / λ", ylabel="Intensity",
             title="Case 1: Flat → Parallel")
    plot!(p, xs / wv, vec(I_num), label="Numerical", linewidth=2, linestyle=:dash, color=:red)
    savefig(p, "case1_flat_parallel.png")
    println("Saved case1_flat_parallel.png")

    err = norm(I_num - I_exact) / norm(I_exact)
    println("Relative intensity error: $err")
    println()
    return p
end

# ------------------------------------------------------------------
# Case 2: Flat propagation (ny = 1) to tilted plane (theta = 30°)
# ------------------------------------------------------------------

function case2_flat_tilted()
    println("=== Case 2: Flat (ny=1) to tilted plane (θ=30°) ===")

    span_x = 200 * wv
    nx = 401
    dx_ = span_x / (nx - 1)

    pl_source = Plane([0.0, 0.0, 0.0];
                      span=[span_x, wv],
                      num_nodes=[nx, 1],
                      dtype=Float64)

    xs = collect(range(-span_x/2, span_x/2; length=nx))
    u_source = exact_field(xs, [0.0], 0.0)

    pl_target = Plane([0.0, 0.0, prop_dist];
                      span=[span_x, wv],
                      num_nodes=[nx, 1],
                      orientation=[deg2rad(30.0), 0.0],
                      dtype=Float64)

    dxi, _ = estimate_dxi_deta(pl_target, wv, dx_, dx_)
    println("  Estimated dxi = $dxi")
    wf = Wavefront(FromField(), pl_source, u_source, dxi, 1.0)
    println("  Actual dxi = $(wf.dxi), nxi = $(length(wf.xi))")

    u_num = propagate(wf, wv, pl_target)

    it = xwise_iterator(pl_target, 1)
    u_exact_target = zeros(ComplexF64, 1, nx)
    for (xs_g, ys_g, zs_g) in it
        for j in 1:nx
            u_exact_target[1, j] = (point_source_field(xs_g[j], ys_g[j], zs_g[j], source1_pos...) +
                                    point_source_field(xs_g[j], ys_g[j], zs_g[j], source2_pos...))
        end
    end

    I_num = abs2.(u_num)
    I_exact = abs2.(u_exact_target)

    p = plot(xs / wv, vec(I_exact), label="Exact", linewidth=2, color=:blue,
             xlabel="x / λ", ylabel="Intensity",
             title="Case 2: Flat → Tilted (θ=30°)")
    plot!(p, xs / wv, vec(I_num), label="Numerical", linewidth=2, linestyle=:dash, color=:red)
    savefig(p, "case2_flat_tilted.png")
    println("Saved case2_flat_tilted.png")

    err = norm(I_num - I_exact) / norm(I_exact)
    println("Relative intensity error: $err")
    println()
    return p
end

# ------------------------------------------------------------------
# Case 3: 3D propagation (ny > 1) to parallel plane
# ------------------------------------------------------------------

function case3_3d_parallel()
    println("=== Case 3: 3D to parallel plane ===")

    span_x = span_y = 100 * wv
    nx = ny = 51
    dx_ = span_x / (nx - 1)
    dy_ = span_y / (ny - 1)

    pl_source = Plane([0.0, 0.0, 0.0];
                      span=[span_x, span_y],
                      num_nodes=[nx, ny],
                      dtype=Float64)

    xs = collect(range(-span_x/2, span_x/2; length=nx))
    ys = collect(range(-span_y/2, span_y/2; length=ny))
    u_source = exact_field(xs, ys, 0.0)

    pl_target = Plane([0.0, 0.0, prop_dist];
                      span=[span_x, span_y],
                      num_nodes=[nx, ny],
                      dtype=Float64)

    dxi_nq, deta_nq = estimate_dxi_deta(pl_target, wv, dx_, dy_)
    # Cap nxi to 2*nx for computational feasibility
    dxi = max(dxi_nq, 1.0 / (2 * nx * dx_))
    deta = max(deta_nq, 1.0 / (2 * ny * dy_))
    println("  Nyquist dxi = $dxi_nq, capped dxi = $dxi")
    println("  Nyquist deta = $deta_nq, capped deta = $deta")
    wf = Wavefront(FromField(), pl_source, u_source, dxi, deta)
    println("  Actual dxi = $(wf.dxi), nxi = $(length(wf.xi))")
    println("  Actual deta = $(wf.deta), neta = $(length(wf.eta))")

    u_num = propagate_parallel(wf, wv, pl_target)
    u_exact_target = exact_field(xs, ys, prop_dist)

    I_num = abs2.(u_num)
    I_exact = abs2.(u_exact_target)
    I_diff = abs.(I_num - I_exact)

    vmax = maximum(I_exact)

    p1 = heatmap(xs / wv, ys / wv, I_exact', clim=(0, vmax), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Exact",
                 colorbar_title="Intensity")
    p2 = heatmap(xs / wv, ys / wv, I_num', clim=(0, vmax), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Numerical",
                 colorbar_title="Intensity")
    p3 = heatmap(xs / wv, ys / wv, I_diff', clim=(0, vmax*0.5), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Difference",
                 colorbar_title="|ΔI|")

    p = plot(p1, p2, p3, layout=(1, 3), size=(1500, 400))
    savefig(p, "case3_3d_parallel.png")
    println("Saved case3_3d_parallel.png")

    err = norm(I_num - I_exact) / norm(I_exact)
    println("Relative intensity error: $err")
    println()
    return p
end

# ------------------------------------------------------------------
# Case 4: 3D propagation (ny > 1) to tilted plane (theta = 30°)
# ------------------------------------------------------------------

function case4_3d_tilted()
    println("=== Case 4: 3D to tilted plane (θ=30°) ===")

    span_x = span_y = 100 * wv
    nx = ny = 51
    dx_ = span_x / (nx - 1)
    dy_ = span_y / (ny - 1)

    pl_source = Plane([0.0, 0.0, 0.0];
                      span=[span_x, span_y],
                      num_nodes=[nx, ny],
                      dtype=Float64)

    xs = collect(range(-span_x/2, span_x/2; length=nx))
    ys = collect(range(-span_y/2, span_y/2; length=ny))
    u_source = exact_field(xs, ys, 0.0)

    pl_target = Plane([0.0, 0.0, prop_dist];
                      span=[span_x, span_y],
                      num_nodes=[nx, ny],
                      orientation=[deg2rad(30.0), 0.0],
                      dtype=Float64)

    dxi_nq, deta_nq = estimate_dxi_deta(pl_target, wv, dx_, dy_)
    dxi = max(dxi_nq, 1.0 / (2 * nx * dx_))
    deta = max(deta_nq, 1.0 / (2 * ny * dy_))
    println("  Nyquist dxi = $dxi_nq, capped dxi = $dxi")
    println("  Nyquist deta = $deta_nq, capped deta = $deta")
    wf = Wavefront(FromField(), pl_source, u_source, dxi, deta)
    println("  Actual dxi = $(wf.dxi), nxi = $(length(wf.xi))")
    println("  Actual deta = $(wf.deta), neta = $(length(wf.eta))")

    u_num = propagate(wf, wv, pl_target; max_mem_mb=8192)

    it = xwise_iterator(pl_target, ny)
    u_exact_target = zeros(ComplexF64, ny, nx)
    row = 1
    for (xs_g, ys_g, zs_g) in it
        sy_eff = size(xs_g, 2)
        for j in 1:nx, l in 1:sy_eff
            u_exact_target[row + l - 1, j] = (point_source_field(xs_g[j, l], ys_g[j, l], zs_g[j, l], source1_pos...) +
                                              point_source_field(xs_g[j, l], ys_g[j, l], zs_g[j, l], source2_pos...))
        end
        row += sy_eff
    end

    I_num = abs2.(u_num)
    I_exact = abs2.(u_exact_target)
    I_diff = abs.(I_num - I_exact)

    vmax = maximum(I_exact)

    p1 = heatmap(xs / wv, ys / wv, I_exact', clim=(0, vmax), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Exact",
                 colorbar_title="Intensity")
    p2 = heatmap(xs / wv, ys / wv, I_num', clim=(0, vmax), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Numerical",
                 colorbar_title="Intensity")
    p3 = heatmap(xs / wv, ys / wv, I_diff', clim=(0, vmax*0.5), color=:viridis,
                 xlabel="x / λ", ylabel="y / λ", title="Difference",
                 colorbar_title="|ΔI|")

    p = plot(p1, p2, p3, layout=(1, 3), size=(1500, 400))
    savefig(p, "case4_3d_tilted.png")
    println("Saved case4_3d_tilted.png")

    err = norm(I_num - I_exact) / norm(I_exact)
    println("Relative intensity error: $err")
    println()
    return p
end

# ------------------------------------------------------------------
# Run all cases
# ------------------------------------------------------------------

gr()  # Use GR backend

case1_flat_parallel()
case2_flat_tilted()
case3_3d_parallel()
case4_3d_tilted()

println("All cases completed!")

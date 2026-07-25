import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

ENV["GKSwstype"] = "100"

using Plots
using LinearAlgebra
include("../src/Waveq.jl")
using .Waveq: Plane, to_plane, from_plane

pl1 = Plane([0.0, 0.0, 0.0]; span=(2.0, 1.5), num_nodes=(4, 3), orientation=(0.0, 0.0))

theta = deg2rad(15.0)
phi = deg2rad(45.0)
pl2 = Plane([0.0, 0.0, 1.0]; span=(2.0, 1.5), num_nodes=(4, 3), orientation=(theta, phi))

# Generate grid points
function get_points(pl::Plane)
    nx, ny = pl.num_nodes[1], pl.num_nodes[2]
    pts = zeros(ny, nx, 3)
    for j in 1:ny
        yl = -pl.span[2]/2 + (j - 1) * pl.dy_
        for i in 1:nx
            xl = -pl.span[1]/2 + (i - 1) * pl.dx_
            g = pl.origin .+ xl .* pl.ort_x .+ yl .* pl.ort_y
            pts[j, i, :] .= g
        end
    end
    return pts
end

p1_grid = get_points(pl1)
p2_grid = get_points(pl2)

# Verify round-trip transforms
local1 = to_plane(pl1, p1_grid)
restored1 = from_plane(pl1, local1)
@assert isapprox(restored1, p1_grid)
@assert isapprox(local1[:, :, 3], zeros(3, 4), atol=1e-12)

local2 = to_plane(pl2, p2_grid)
restored2 = from_plane(pl2, local2)
@assert isapprox(restored2, p2_grid)
@assert isapprox(local2[:, :, 3], zeros(3, 4), atol=1e-12)

println("Round-trip transforms OK")

nx, ny = pl1.num_nodes[1], pl1.num_nodes[2]
corners = Dict(
    "A" => (1, 1),
    "B" => (1, nx),
    "C" => (ny, nx),
    "D" => (ny, 1),
)

# Plotting functions
function get_proj(p_matrix, proj_type)
    # p_matrix is (ny, nx, 3)
    p_flat = reshape(p_matrix, (:, 3))
    if proj_type == :xy
        return p_flat[:, 1], p_flat[:, 2]
    elseif proj_type == :xz
        return p_flat[:, 1], p_flat[:, 3]
    elseif proj_type == :yz
        return p_flat[:, 2], p_flat[:, 3]
    end
end

function plot_proj(proj_type, xlabel_str, ylabel_str, title_str)
    x1, y1 = get_proj(p1_grid, proj_type)
    x2, y2 = get_proj(p2_grid, proj_type)
    
    p = scatter(x1, y1, c=:steelblue, m=:circle, ms=6, label="pl1 (z=0)", aspect_ratio=:equal, frame=:box)
    scatter!(p, x2, y2, c=:coral, m=:square, ms=6, label="pl2 (θ=15°, φ=45°)")
    
    for (name, (j, i)) in corners
        pt1 = p1_grid[j, i, :]
        pt2 = p2_grid[j, i, :]
        if proj_type == :xy
            annotate!(p, pt1[1], pt1[2], text(name, 10, :steelblue, :bottom))
            annotate!(p, pt2[1], pt2[2], text(name * "′", 10, :coral, :bottom))
        elseif proj_type == :xz
            annotate!(p, pt1[1], pt1[3], text(name, 10, :steelblue, :bottom))
            annotate!(p, pt2[1], pt2[3], text(name * "′", 10, :coral, :bottom))
        elseif proj_type == :yz
            annotate!(p, pt1[2], pt1[3], text(name, 10, :steelblue, :bottom))
            annotate!(p, pt2[2], pt2[3], text(name * "′", 10, :coral, :bottom))
        end
    end
    xlabel!(p, xlabel_str)
    ylabel!(p, ylabel_str)
    title!(p, title_str)
    return p
end

p_xy = plot_proj(:xy, "x", "y", "XY projection")
p_xz = plot_proj(:xz, "x", "z", "XZ projection")
p_yz = plot_proj(:yz, "y", "z", "YZ projection")

# 3D view
p1_flat = reshape(p1_grid, (:, 3))
p2_flat = reshape(p2_grid, (:, 3))
p_3d = scatter(p1_flat[:,1], p1_flat[:,2], p1_flat[:,3], c=:steelblue, m=:circle, ms=6, label="pl1 (z=0)", legend=:topleft, camera=(30, 30))
scatter!(p_3d, p2_flat[:,1], p2_flat[:,2], p2_flat[:,3], c=:coral, m=:square, ms=6, label="pl2 (θ=15°, φ=45°)")

for (name, (j, i)) in corners
    pt1 = p1_grid[j, i, :]
    pt2 = p2_grid[j, i, :]
    annotate!(p_3d, pt1[1], pt1[2], pt1[3], text(name, 10, :steelblue))
    annotate!(p_3d, pt2[1], pt2[2], pt2[3], text(name * "′", 10, :coral))
end
xlabel!(p_3d, "x")
ylabel!(p_3d, "y")
zlabel!(p_3d, "z")
title!(p_3d, "Isometric view")

fig = plot(p_xy, p_xz, p_yz, p_3d, layout=(2, 2), size=(1000, 1000), plot_title="3-D Plane Coordinate Transform Verification")

savefig(fig, joinpath(@__DIR__, "plane_transform_verify.pdf"))
println("Saved plane_transform_verify.pdf")

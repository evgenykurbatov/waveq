module glass

export Glass, refractive_index,
       wv_ref_F, wv_ref_e, wv_ref_d, wv_ref_C,
       N_BK7, F2, SF5

const wv_ref_F = 0.48613  # [um], H blue line
const wv_ref_e = 0.54607  # [um], Hg green line
const wv_ref_d = 0.58756  # [um], He yellow line
const wv_ref_C = 0.65627  # [um], H red line

struct Glass
    name::String
    wv_min::Float64
    wv_max::Float64
    B::Vector{Float64}
    C::Vector{Float64}
end

function refractive_index(g::Glass, wv::Real)::Float64
    return sqrt(1.0 + sum(g.B .* wv^2 ./ (wv^2 .- g.C)))
end

const N_BK7 = Glass(
    "N-BK7",
    0.365,
    2.5,
    [1.03961212, 0.231792344, 1.01046945],
    [0.00600069867, 0.0200179144, 103.560653],
)

const F2 = Glass(
    "F2",
    0.365,
    2.5,
    [1.34533359, 0.209073176, 0.937357162],
    [0.00997743871, 0.0470450767, 111.886764],
)

const SF5 = Glass(
    "SF5",
    0.365,
    2.5,
    [1.52481889, 0.187085527, 1.42729015],
    [0.011254756, 0.0588995392, 129.141675],
)

end # module

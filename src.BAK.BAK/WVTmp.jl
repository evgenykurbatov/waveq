module Waveq

export Plane, PointIterator, XwiseIterator, YwiseIterator,
       to_plane, from_plane, from_plane2d,
       point_iterator, xwise_iterator, ywise_iterator,
       Glass, refractive_index,
       wv_ref_F, wv_ref_e, wv_ref_d, wv_ref_C,
       N_BK7, F2, SF5,
       nyquist_as_res, nyquist_as_res_alt,
        direct_ft2, czt_ft2, iczt_ft2,
        Wavefront, FromSpectrum, FromField, FromIntensity, propagate, propagate_parallel

include("plane.jl")
include("glass.jl")
include("fourier.jl")
include("scalarfront.jl")

import .plane: Plane, PointIterator, XwiseIterator, YwiseIterator,
       to_plane, from_plane, from_plane2d,
       point_iterator, xwise_iterator, ywise_iterator
import .glass: Glass, refractive_index,
       wv_ref_F, wv_ref_e, wv_ref_d, wv_ref_C,
       N_BK7, F2, SF5
import .fourier: direct_ft2, czt_ft2, iczt_ft2
import .scalarfront: nyquist_as_res, nyquist_as_res_alt,
       Wavefront, FromSpectrum, FromField, FromIntensity, propagate, propagate_parallel

end # module

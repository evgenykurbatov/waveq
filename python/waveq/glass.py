import os
import numpy as np

_JULIA_PROJECT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "Waveq")
_JULIA_PROJECT = os.path.abspath(_JULIA_PROJECT)
os.environ.setdefault("JULIA_PROJECT", _JULIA_PROJECT)

from juliacall import Main as jl


class Glass:
    def __init__(self, jl_glass):
        self._jl = jl_glass

    @property
    def name(self):
        return str(self._jl.name)

    @property
    def wv_min(self):
        return float(self._jl.wv_min)

    @property
    def wv_max(self):
        return float(self._jl.wv_max)

    def refractive_index(self, wv):
        return float(jl.Waveq.refractive_index(self._jl, float(wv)))

    def refractive_index_array(self, wv_array):
        return np.array([self.refractive_index(w) for w in wv_array])


N_BK7 = Glass(jl.Waveq.N_BK7)
F2 = Glass(jl.Waveq.F2)
SF5 = Glass(jl.Waveq.SF5)

wv_ref_F = float(jl.Waveq.wv_ref_F)
wv_ref_e = float(jl.Waveq.wv_ref_e)
wv_ref_d = float(jl.Waveq.wv_ref_d)
wv_ref_C = float(jl.Waveq.wv_ref_C)

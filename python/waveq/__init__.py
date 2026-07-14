"""
waveq - Wave optics calculations via Julia

Python wrapper around the Waveq Julia package.
"""

import os
import sys

_JULIA_PROJECT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "Waveq")
_JULIA_PROJECT = os.path.abspath(_JULIA_PROJECT)
os.environ.setdefault("JULIA_PROJECT", _JULIA_PROJECT)

from juliacall import Main as jl

jl.seval(f'using Pkg; Pkg.activate("{_JULIA_PROJECT}")')
jl.seval("using Waveq")

import numpy as np


from .plane import Plane
from .glass import Glass, N_BK7, F2, SF5, wv_ref_F, wv_ref_e, wv_ref_d, wv_ref_C


def nyquist_as_res(wv, x, y, z, dx, dy):
    result = jl.Waveq.nyquist_as_res(
        float(wv),
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        float(dx),
        float(dy),
    )
    return float(result[0]), float(result[1])


def nyquist_as_res_alt(wv, p, dx, dy):
    result = jl.Waveq.nyquist_as_res_alt(
        float(wv),
        np.asarray(p, dtype=np.float64),
        float(dx),
        float(dy),
    )
    return float(result[0]), float(result[1])


def direct_ft2(f, x_, y_, xi, eta):
    result = jl.Waveq.direct_ft2(
        np.asarray(f, dtype=np.complex128),
        np.asarray(x_, dtype=np.float64),
        np.asarray(y_, dtype=np.float64),
        np.asarray(xi, dtype=np.float64),
        np.asarray(eta, dtype=np.float64),
    )
    return np.array(result)


def czt_ft2(f, x_, y_, nxi, neta, dx=None, dy=None):
    kwargs = {}
    if dx is not None:
        kwargs["dx"] = float(dx)
    if dy is not None:
        kwargs["dy"] = float(dy)
    result = jl.Waveq.czt_ft2(
        np.asarray(f, dtype=np.complex128),
        np.asarray(x_, dtype=np.float64),
        np.asarray(y_, dtype=np.float64),
        int(nxi),
        int(neta),
        **kwargs,
    )
    F = np.array(result[0])
    xi = np.array(result[1])
    eta = np.array(result[2])
    return F, xi, eta


FromSpectrum = jl.Waveq.FromSpectrum
FromField = jl.Waveq.FromField
FromIntensity = jl.Waveq.FromIntensity


class Wavefront:
    def __init__(self, source, pl1, data, dxi, deta):
        jl_pl1 = pl1._jl if hasattr(pl1, "_jl") else pl1
        self._jl = jl.Waveq.Wavefront(
            source,
            jl_pl1,
            np.asarray(data),
            float(dxi),
            float(deta),
        )

    @property
    def U(self):
        return np.array(self._jl.U)

    @property
    def xi(self):
        return np.array(self._jl.xi)

    @property
    def eta(self):
        return np.array(self._jl.eta)

    @property
    def dxi(self):
        return float(self._jl.dxi)

    @property
    def deta(self):
        return float(self._jl.deta)

    @property
    def u(self):
        u = self._jl.u
        return np.array(u) if u is not None else None


def propagate(wf, wv, pl2):
    jl_pl2 = pl2._jl if hasattr(pl2, "_jl") else pl2
    return jl.Waveq.propagate(wf._jl, float(wv), jl_pl2)


__all__ = ["Plane",
           "Glass", "N_BK7", "F2", "SF5",
           "wv_ref_F", "wv_ref_e", "wv_ref_d", "wv_ref_C",
           "nyquist_as_res", "nyquist_as_res_alt",
           "direct_ft2", "czt_ft2",
           "Wavefront", "FromSpectrum", "FromField", "FromIntensity",
           "propagate"]

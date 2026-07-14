import os
import numpy as np

_JULIA_PROJECT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "Waveq")
_JULIA_PROJECT = os.path.abspath(_JULIA_PROJECT)
os.environ.setdefault("JULIA_PROJECT", _JULIA_PROJECT)

from juliacall import Main as jl


class Plane:
    def __init__(self, origin, span=None, num_nodes=None, resolution=None,
                 orientation=None, dtype=np.float64):
        self.dtype = dtype
        kwargs = {}
        if span is not None:
            kwargs["span"] = np.asarray(span, dtype=dtype)
        if num_nodes is not None:
            kwargs["num_nodes"] = np.asarray(num_nodes, dtype=int)
        if resolution is not None:
            kwargs["resolution"] = np.asarray(resolution, dtype=dtype)
        if orientation is not None:
            kwargs["orientation"] = np.asarray(orientation, dtype=dtype)
        self._jl = jl.Waveq.Plane(np.asarray(origin, dtype=dtype), **kwargs)

    @property
    def origin(self):
        return np.array(self._jl.origin)

    @property
    def span(self):
        return np.array(self._jl.span)

    @property
    def num_nodes(self):
        return [int(self._jl.num_nodes[0]), int(self._jl.num_nodes[1])]

    @property
    def orientation(self):
        return np.array(self._jl.orientation)

    @property
    def ort_x(self):
        return np.array(self._jl.ort_x)

    @property
    def ort_y(self):
        return np.array(self._jl.ort_y)

    @property
    def ort_norm(self):
        return np.array(self._jl.ort_norm)

    @property
    def dx_(self):
        return float(self._jl.dx_)

    @property
    def dy_(self):
        return float(self._jl.dy_)

    @property
    def p(self):
        nx, ny = self.num_nodes
        x_ = np.linspace(-self.span[0] / 2, self.span[0] / 2, nx, dtype=self.dtype)
        y_ = np.linspace(-self.span[1] / 2, self.span[1] / 2, ny, dtype=self.dtype)
        X_, Y_ = np.meshgrid(x_, y_, indexing="xy")
        return self.origin + X_[..., np.newaxis] * self.ort_x + Y_[..., np.newaxis] * self.ort_y

    def to_plane(self, p):
        return np.array(jl.Waveq.to_plane(self._jl, np.asarray(p, dtype=self.dtype)))

    def from_plane(self, p_):
        return np.array(jl.Waveq.from_plane(self._jl, np.asarray(p_, dtype=self.dtype)))

    def from_plane2d(self, p_):
        return np.array(jl.Waveq.from_plane2d(self._jl, np.asarray(p_, dtype=self.dtype)))

    def point_iterator(self):
        it = jl.Waveq.point_iterator(self._jl)
        for x, y, z in it:
            yield np.array(x), np.array(y), np.array(z)

    def xwise_iterator(self, sy):
        it = jl.Waveq.xwise_iterator(self._jl, sy)
        for x, y, z in it:
            yield np.array(x), np.array(y), np.array(z)

    def ywise_iterator(self, sx):
        it = jl.Waveq.ywise_iterator(self._jl, sx)
        for x, y, z in it:
            yield np.array(x), np.array(y), np.array(z)

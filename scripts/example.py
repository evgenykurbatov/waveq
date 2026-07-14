#!/usr/bin/env python3
"""
Example: Using waveq from Python via juliacall

Run with:
    pixi run python scripts/example.py
"""

import os

os.environ["PYTHON_JULIACALL_THREADS"] = "auto"

import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from waveq import Plane, N_BK7, nyquist_as_res, direct_ft2, czt_ft2

print("Python version:", sys.version)
print()

# Build two planes
pl1 = Plane(
    origin=[0.0, 0.0, 0.0],
    span=[2.0, 1.5],
    num_nodes=[4, 3],
    orientation=[0.0, 0.0],
    dtype=np.float64,
)

pl2 = Plane(
    origin=[0.0, 0.0, 1.0],
    span=[2.0, 1.5],
    num_nodes=[4, 3],
    orientation=[np.deg2rad(15.0), np.deg2rad(45.0)],
    dtype=np.float64,
)

print("pl1 p shape:", pl1.p.shape)
print("pl2 p shape:", pl2.p.shape)

# Test Nyquist sampling
x = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
y = np.array([0.0, 0.3, -0.3, 0.8, -0.8])
z = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
dxi, deta = nyquist_as_res(632.8e-9, x, y, z, 10e-6, 10e-6)
print(f"Nyquist dxi = {dxi:.6f}, deta = {deta:.6f}")

# Test CZT
f = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
x_ = np.linspace(-1.0, 1.0, 64)
y_ = np.linspace(-1.0, 1.0, 64)
F, xi, eta = czt_ft2(f, x_, y_, 51, 51)
print(f"CZT output shape: {F.shape}")
print(f"xi range: {xi[0]:.4f} to {xi[-1]:.4f}")
print(f"eta range: {eta[0]:.4f} to {eta[-1]:.4f}")

print("\nAll OK")

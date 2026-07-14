# Heavy Computations in Waveq: Propagation & Fourier Transforms

This document collects all performance-related considerations for the angular-spectrum propagation and CZT-based Fourier transforms in the `Waveq` Julia package.

---

## 1. Direct Propagation: Stripe-Based Matrix-Vector Approach

### 1.1 Mathematical formulation

The propagation integral from the angular spectrum $U(\xi, \eta)$ to an arbitrarily-tilted plane $(x, y)$ is:

$$
u(x, y) = \Delta\xi \Delta\eta \sum_{p,q} e^{i 2\pi (\xi_p x + \eta_q y)} \, e^{i \kappa(\xi_p, \eta_q) z} \, U_{pq}
$$

where $z = z(x, y)$ because the target plane is tilted relative to the source. The longitudinal wavenumber is:

$$
\kappa(\xi, \eta) = \begin{cases}
2\pi \sqrt{\lambda^{-2} - \xi^2 - \eta^2} & \text{(propagating)} \\
2\pi i \sqrt{\xi^2 + \eta^2 - \lambda^{-2}} & \text{(evanescent)}
\end{cases}
$$

### 1.2 Stripe decomposition

The computation is split by **stripes** (contiguous $y$-slices) on the target plane. For a stripe of height $sy$:

- Target points: $nx \times sy$
- Spectrum size: $neta \times nxi$
- Phase tensor $A$: shape $(nx, neta, nxi, sy)$
- Reshaped to $A_{block}$: shape $(nx \cdot sy) \times (neta \cdot nxi)$
- Single BLAS `gemv`: $u = A_{block} \cdot \text{vec}(U)$

Stripe height $sy$ is chosen so that $A_{block}$ fits in `max_mem_mb` (default 2 GB).

### 1.3 Why this is $O(N^4)$

For source grid $(nxi, neta)$ and target grid $(nx, ny)$:

$$
\text{Total FLOPs} \approx 2 \cdot nx \cdot ny \cdot nxi \cdot neta
$$

This is the exact cost of a dense matrix-vector product over the full 4D index space. There is no FFT shortcut because $z$ varies with $x$ and $y$ (tilted plane breaks separability).

---

## 2. CZT Algorithm Cost Analysis

### 2.1 Bluestein's algorithm

The Chirp Z-Transform (CZT) evaluates a DFT at arbitrary frequency points using FFT convolution. For input length $N$ and output length $M$:

| Step | Cost |
|---|---|
| Chirp pre-multiply | $O(N)$ |
| Build chirp filter | $O(N+M)$ |
| FFT convolution (pad to $L \geq N+M-1$) | $O(L \log L)$ |
| Post-multiply | $O(M)$ |

**Total:** $O(L \log L)$ where $L = \text{nextpow2}(N+M-1) \approx 2(N+M)$.

### 2.2 Cached chirp (amortized cost)

When the same $(N, M, \alpha)$ is reused — as in `czt_ft2` processing many columns with identical grid geometry — the FFT of the chirp filter $h$ is computed once:

**Cached cost:** $O(L \log L) + O(N+M) \approx O\bigl((N+M)\log(N+M)\bigr)$.

### 2.3 2D CZT in `czt_ft2`

```julia
F = czt(f, (1.0, 1.0), (1, 2), (neta, nxi))
```

- Separable CZT along rows (dimension 1, $y/\eta$) and columns (dimension 2, $x/\xi$)
- Cost: $O(nx \cdot neta \log neta + ny \cdot nxi \log nxi) \approx O(N^2 \log N)$ for square grids
- Constant factor: ~2–4× slower than standard FFT of the same size due to padding and extra multiplies

### 2.4 Dummy dimensions ($N=1$)

When $nx = 1$ (dummy $x$), CZT with scale=1.0 maps the single DC component correctly. The pre-multiplied chirp of a single sample produces a constant output scaled appropriately, so dummy dimensions are handled naturally.

---

## 3. Arithmetic Intensity & Memory Bottlenecks

### 3.1 Intensity of direct propagation

For the `gemv` step per stripe:

- Matrix size: $M = nx \cdot sy$ rows, $N = neta \cdot nxi$ columns
- FLOPs: $2 \cdot M \cdot N$ (1 multiply + 1 add per element)
- Memory read: $M \cdot N$ complex words for $A_{block}$ + $N$ for $\text{vec}(U)$ + $M$ for output
- **Arithmetic intensity ≈ 2**

This is **memory-bound**. Peak performance is limited by memory bandwidth, not compute.

### 3.2 Memory footprint per stripe

```
A_block  = nx * sy * neta * nxi * sizeof(Complex{T})
phase_x  = nx * nxi * sy * sizeof(Complex{T})
phase_y  = nx * neta * sy * sizeof(Complex{T})
phase_z  = nx * neta * nxi * sy * sizeof(Complex{T})  -- fused, not stored separately
U        = neta * nxi * sizeof(Complex{T})
u_stripe = nx * sy * sizeof(Complex{T})
```

With `max_mem_mb`, the dominant temporary `A_block` is constrained. For `ComplexF64`:

| Grid size | `sy=1` A_block size | `sy=16` A_block size |
|---|---|---|
| 64³ | 32 MB | 512 MB |
| 128³ | 256 MB | 4 GB |
| 256³ | 2 GB | 32 GB (needs tiling) |
| 512³ | 16 GB | 128 GB (needs tiling) |

---

## 4. CPU Multicore Performance

### 4.1 Single-core performance

| Scenario | Typical throughput | Bottleneck |
|---|---|---|
| `nx=256, nxi=256, neta=256, sy=1` | ~2–5 GFLOP/s | Memory bandwidth (~20 GB/s) |
| `nx=512, nxi=512, neta=512, sy=1` | ~3–6 GFLOP/s | Memory bandwidth |

### 4.2 Multicore scaling

| Cores | Efficiency | Notes |
|---|---|---|
| 2 | ~1.8× | Good, independent memory channels |
| 4 | ~3.2× | Moderate, shared LLC pressure |
| 8 | ~5× | Diminishing, memory bandwidth wall |
| 16+ | ~6–8× | Poor — DDR4 bandwidth saturated |

**Key insight:** Because arithmetic intensity ≈ 2, adding cores beyond 4–8 yields minimal returns on typical DDR4 desktop systems (50 GB/s → ~25 GFLOP/s peak in complex double precision).

### 4.3 Better parallelization: independent stripes

Instead of relying on BLAS threading for a single `gemv`, process **multiple stripes in parallel** (e.g. `@threads` over stripe batches). This:
- Avoids cache contention
- Uses cores on independent data
- Scales linearly with stripe count up to memory bandwidth limits

```julia
# Pseudocode for multi-stripe parallelization
@threads for batch in stripe_batches
    for stripe in batch
        # Each thread computes its own A_block and gemv
        u_out[y_start:y_end, :] .= compute_stripe(...)
    end
end
```

---

## 5. GPU (CUDA) Performance

### 5.1 GPU throughput estimates

| Scenario | Typical throughput | Bottleneck |
|---|---|---|
| `nx=512, nxi=512, neta=512, sy=1` on RTX 4090 | ~100–300 GFLOP/s | Memory bandwidth (~1 TB/s) |
| `nx=512, sy=4` | ~300–600 GFLOP/s | Better GPU utilization |
| `nx=64, sy=1` (small grid) | Poor | Kernel launch overhead dominates |
| `nx=1024, nxi=1024, neta=1024, sy=1` on A100 | ~1–2 TFLOP/s | HBM2e bandwidth (~2 TB/s) |

### 5.2 GPU speedup vs. CPU

| Grid size | Expected GPU speedup (vs. single-core) | Notes |
|---|---|---|
| 64³ | 1–2× | Overhead, not worth it |
| 128³ | 5–10× | Good utilization |
| 256³ | 10–20× | Clear winner |
| 512³ | 15–30× | Bandwidth-limited on both |
| 1024³ | 20–50× | Requires 24+ GB GPU memory or tiling |

### 5.3 Julia + CUDA.jl integration

Julia's broadcasting and BLAS dispatch automatically to cuBLAS when inputs are `CuArray`:

| Operation | CPU (`Array`) | GPU (`CuArray`) |
|---|---|---|
| `cis.(...)` | CPU SIMD loop | CUDA kernel launch |
| `reshape` | View (no copy) | View (no copy) |
| `.*` (broadcast) | CPU loop | CUDA fused kernel |
| `A_block * vec(U)` | OpenBLAS/MKL `gemv` | cuBLAS `cgemv` |
| `Array(u_out)` | — | Device→Host `cudaMemcpy` |

**No manual kernel writing required.** The code changes are minimal:

```julia
if use_gpu
    U_gpu  = CuArray(wf.U)
    xi_gpu = CuArray(wf.xi)
    eta_gpu = CuArray(wf.eta)
    K_gpu  = CuArray(K_host)
    u_out  = CuArray{Complex{T}}(undef, ny2, nx2)
else
    U_gpu, xi_gpu, eta_gpu, K_gpu = wf.U, wf.xi, wf.eta, K_host
    u_out = Matrix{Complex{T}}(undef, ny2, nx2)
end
```

### 5.4 Memory capacity limits

For `ComplexF64` (16 bytes per element), the `A_block` size for one stripe is:

```
A_block_bytes = nx * sy * neta * nxi * 16
```

| Grid | `A_block` (`sy=1`) | Fits on RTX 4090 (24 GB)? | Fits on A100 (80 GB)? |
|---|---|---|---|
| 256³ | 2 GB | ✅ Yes | ✅ Yes |
| 512³ | 16 GB | ✅ Yes | ✅ Yes |
| 1024³ | 128 GB | ❌ No | ❌ No |
| 2048³ | 1024 GB | ❌ No | ❌ No |

For grids exceeding GPU memory, **tile the computation** by processing sub-blocks of the spectrum or sub-chunks of the target grid.

### 5.5 Multi-GPU / tiling strategy

When a single `A_block` for `sy=1` exceeds GPU memory:

| Tiling dimension | Memory formula | Overhead |
|---|---|---|
| Tile spectrum `(nxi, neta)` into sub-blocks | `nx·sy·neta_tile·nxi_tile` | **None** — exact |
| Tile target `nx` into column chunks | `nx_chunk·sy·neta·nxi` | **None** — exact |

Both preserve the exact direct sum while respecting memory limits. The `max_mem_mb` parameter in `propagate()` already implements automatic stripe-height reduction as a form of target tiling.

---

## 6. Comparison: Exact Direct vs. CZT/FFT vs. Approximate Methods

### 6.1 Exact methods

| Method | Cost | Correct for tilted planes? |
|---|---|---|
| **Direct summation (our implementation)** | $O(nx \cdot ny \cdot nxi \cdot neta)$ | **Yes** — exact |
| **CZT/FFT per fixed-$z$ slice** | $O(nx \cdot ny \log(nx \cdot ny))$ per slice | Only if $z = \text{const}$ |
| **Separable CZT** | $O(N^2 \log N)$ | **No** — requires $z = \text{const}$ |

### 6.2 Approximate fast methods

#### Method A: Reference plane + residual phase mask

1. Propagate to parallel reference plane $z = z_0$ via CZT: $O(N^2 \log N)$
2. Apply phase correction $e^{i k_0 \delta z}$ and Jacobian factor: $O(N^2)$

**Valid when:** $\frac{\pi w \tan\theta}{\lambda} \cdot \text{NA}^2 \ll 1$

**Cost:** $O(N^2 \log N)$, ~2–4× slower than FFT.

#### Method B: Multi-plane interpolation

1. Choose $N_z$ parallel planes spanning the $z$-range
2. Compute $u_k(x, y)$ via CZT for each: $O(N_z \cdot N^2 \log N)$
3. Interpolate between nearest $z$-slices per point

**Cost:** $O(N_z \cdot N^2 \log N)$. Captures diffraction evolution without assuming $\delta z$ is small.

#### Method C: Paraxial spectral rotation

In the paraxial limit $\kappa \approx k_0 - \frac{\pi}{k_0}(\xi^2 + \eta^2)$, the tilted-plane exponent becomes approximately separable with a frequency shift. This is equivalent to a spatial phase ramp.

**Valid when:** Very small tilt angles only.

### 6.3 Summary table

| Method | Cost | Error source | Valid when |
|---|---|---|---|
| **Exact direct sum** | $O(N^4)$ | None (up to sampling) | Always |
| **A: Ref. plane + phase** | $O(N^2 \log N)$ | Neglects diffraction over $\delta z$ | Small tilt, low NA |
| **B: Multi-plane interp.** | $O(N_z N^2 \log N)$ | Interpolation error | Moderate tilt |
| **C: Spectral rotation** | $O(N^2 \log N)$ | Paraxial + small-tilt | Very small tilt only |

---

## 7. Practical Recommendations

### Small grids ($<64^3$)
- **Use CPU**, single-core or few cores
- GPU overhead not worth it
- Exact direct sum is fast enough

### Medium grids ($64^3$ – $256^3$)
- **Use multicore CPU** (4–8 cores) with independent stripe parallelization
- GPU optional, 5–10× speedup
- Exact direct sum still feasible in seconds

### Large grids ($256^3$ – $1024^3$)
- **Use GPU** (8–24 GB VRAM)
- 10–50× speedup vs. CPU
- Monitor memory; tile if `A_block` exceeds VRAM

### Very large grids ($>1024^3$)
- **Multi-GPU or tiling required**
- Consider whether approximate methods (A or B) are acceptable
- Exact direct sum may take minutes even on GPU

### For tilted planes specifically
- **CZT/FFT cannot be used directly** because $z(x, y)$ breaks separability
- **Only exact direct sum is rigorous** for large tilts or high NA
- Approximate methods A/B are viable if tilt is small and NA is low

---

## 8. Implementation Notes

### Current CPU implementation
- `propagate()` in `scalarfront.jl` uses `LinearAlgebra` BLAS (`gemv`)
- Stripe height auto-tuned via `max_mem_mb`
- `xwise_iterator` yields `(xs, ys, zs)` matrices for each stripe
- `to_plane()` transforms global → local coordinates

### GPU migration path
1. Add `CUDA.jl` to `Project.toml`
2. Wrap `wf.U`, `wf.xi`, `wf.eta`, `K_host` in `CuArray(...)`
3. Pre-allocate `u_out` as `CuArray{Complex{T}}`
4. Julia handles the rest via dispatch
5. Copy result back with `Array(u_out)`

### Future optimizations
- **Multi-threaded stripes:** `@threads` over independent stripe batches
- **CUDA streams:** Overlap computation of multiple stripes on GPU
- **Batched GEMV:** Use `CUDA.CUBLAS.gemv_batched` for multiple small stripes
- **Kernel fusion:** Write a custom CUDA kernel that fuses `cis`, broadcast multiply, and sum into a single kernel to reduce memory traffic

---

*Last updated: 2026-07-05*

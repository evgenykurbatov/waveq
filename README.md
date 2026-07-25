<h1 align='center'>Waveq</h1>


**Waveq** is Julia library for Scalar Wave Optics

URL: https://github.com/evgenykurbatov/waveq


## Quick Start

### 1. Install Julia dependencies

```bash
pixi run julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Run examples

```bash
pixi run -e dev julia --project=. examples/A_3d.jl
```


## Author

**Evgeny P. Kurbatov** ([ORCiD](https://orcid.org/0000-0002-1024-9446))

- <evgeny.p.kurbatov@gmail.com>


## References

Projects that inspired me:
- [PyOptica](https://gitlab.com/pyoptica/pyoptica): Diffractive Optics in Python
- [Diffractio](https://github.com/optbrea/diffractio): Python Diffraction-Interference module
- [Waveprop](https://github.com/ebezzam/waveprop): Diffraction-based wave propagation simulator with PyTorch support
- [PyOpticalTable](https://github.com/james-d-pickering/pyopticaltable): Pain-Free Drawing of Optical Setups

Bibliography:
- Goodman J.W. _Introduction to Fourier Optics_. McGraw-Hill. ISBN 0-07-024254-2
- Wang A. _Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction formula_, Applied Optics, 2006, DOI:[10.1364/AO.45.001102](https://doi.org/10.1364/AO.45.001102)
- Matsushima K., Shimobaba T. _Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields_, Optics Express, 2009, DOI:[10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662)
- Matsushima K. _Shifted angular spectrum method for off-axis numerical propagation_, Optics Express, 2010, DOI:[10.1364/OE.18.018453](https://doi.org/10.1364/OE.18.018453)


## Licence

Apache-2.0

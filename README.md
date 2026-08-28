# SLM Top-Hat Beam Shaping

A compact Python toolkit for optimizing SLM holograms that shape an input beam into a top-hat pattern in the focal plane. The code models the SLM plane, focal/camera plane, Fourier propagation, and CG-based hologram optimization.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

If you use `slmsuite` calibration files in the notebooks, install `slmsuite` in the same environment.

## Main Features

- Single-wavelength CG optimization with `CGOptimizer`.
- Dual-wavelength optimization with one shared hologram using `DWCGOptimizer`.
- SLM, camera, and propagation objects with physical focal-plane scaling.
- Rectangle and line top-hat targets with optional PSF smoothing.
- Circular, rectangular, and expanded-support optimization masks.
- Initial holograms including random, zero, lens phase, and curvature phase.
- Superpixel optimization and full-resolution hologram expansion.

## Basic Import

```python
from slm_tophat_beam_shaping import SLMPlane, CameraPlane, Propagator, CGOptimizer
```

## Examples

Example notebooks live in `examples/`:

- `single_wavelength_slmsuite_optimization.ipynb`: one calibration file, one wavelength.
- `dual_wavelength_slmsuite_optimization.ipynb`: two wavelengths sharing one hologram.
- `split_slm_two_beam_optimization.ipynb`: split-SLM workflow for two separated beams.
- `half_slm_parameter_sweep.ipynb`: half-SLM parameter sweep.
- `tophat_scan_experiments.ipynb`: larger scan experiments for fidelity and efficiency.
- `oop_optimizer_workflow.ipynb`: longer development workflow with more intermediate plots.

Old procedural code is kept in `history_implementation/` for reference.

# FlatHat Beam Shaping

FlatHat is a Python workflow for SLM hologram optimization. The current implementation uses explicit optical objects for the SLM plane, focal/camera plane, propagation, and CG optimization. It supports both single-wavelength optimization and dual-wavelength optimization with one shared SLM hologram.

The main use case is shaping an input beam into a flat-top target in the focal plane. The input beam can come from a `slmsuite` wavefront-calibration file, including both measured amplitude and input phase.

## Current Entry Points

Use these notebooks for normal work:

- `single_wavelength_slmsuite_optimization.ipynb`: single-wavelength CG optimization using one slmsuite calibration file.
- `dual_wavelength_slmsuite_optimization.ipynb`: dual-wavelength CG optimization using two slmsuite calibration files and one shared hologram.
- `oop_optimizer_workflow.ipynb`: longer development workflow with more intermediate plots and options.

Old procedural implementations were moved to `history_implementation/` for reference.

## Core Files

```text
optical_planes.py
    OpticalPlane, SLMPlane, CameraPlane

propagator.py
    Propagator and propagation-summary plotting

optimizer.py
    CGOptimizer for one wavelength

dual_wavelength_optimizer.py
    DWCGOptimizer for two wavelengths with one shared hologram

initial_holograms.py
    Initial hologram helpers, including curvature_hologram()

target_profiles.py
    Rectangle/line target generation and PSF smoothing

loss_functions.py
    Overlap metrics, benchmark metrics, and mask builders

propagation_functions.py
    Shifted Fourier transform and focal-plane sampling helpers

array_helpers.py
    Array normalization, cropping/padding, and superpixel helpers

validation_helpers.py
    Shared validation functions
```

## Optical Model

The optical system is represented by three main pieces:

- `SLMPlane`: stores the input beam amplitude and input beam phase on the SLM grid. The optimized hologram is stored separately as `slm.hologram`.
- `CameraPlane`: stores the focal-plane field after propagation.
- `Propagator`: connects one `SLMPlane` to one `CameraPlane` and owns a padded far-field plane used for optimization.

Propagation currently uses a shifted Fourier transform:

```text
farfield = fftshift(fft2(ifftshift(padded_slm_field)))
```

The focal-plane sampling is computed from the SLM physical size:

```text
delta_x = wavelength * focal_length / Lx
delta_y = wavelength * focal_length / Ly
```

where `Lx` and `Ly` are the SLM side lengths after applying the superpixel size.

## Single-Wavelength Optimization

The single-wavelength workflow uses `CGOptimizer`.

Minimal flow:

```python
optimizer = CGOptimizer(
    propagator=propagator,
    target_plane=target_plane,
    target_mask=target_mask,
)
optimizer.set_initial_hologram_array(initial_hologram)
optimizer.optimize(
    maxiter=optimizer_maxiter,
    loss_scale=loss_scale,
    optimize_phase=optimize_phase,
)
result = optimizer.get_result_summary()
```

The loss is based on overlap between the propagated output and the target:

```python
loss = loss_scale * (1 - overlap) ** 2
```

If `optimize_phase=True`, the overlap includes the target phase through `cos(output_phase - target_phase)`. If `False`, only amplitude is optimized.

## Dual-Wavelength Optimization

The dual-wavelength workflow uses `DWCGOptimizer`.

It contains two independent optical channels:

- `propagator_1`, `target_plane_1`, `target_mask_1`
- `propagator_2`, `target_plane_2`, `target_mask_2`

The two channels share one hologram array. Both propagators must use the same SLM shape and `superpixel_size`, but they may have different wavelengths, input fields, camera grids, targets, and masks.

Minimal flow:

```python
optimizer = DWCGOptimizer(
    propagator_1=propagator_1,
    propagator_2=propagator_2,
    target_plane_1=target_plane_1,
    target_mask_1=target_mask_1,
    target_plane_2=target_plane_2,
    target_mask_2=target_mask_2,
)
optimizer.set_initial_hologram_array(initial_hologram)
optimizer.optimize(
    maxiter=optimizer_maxiter,
    loss_scale=loss_scale,
    optimize_phase=optimize_phase,
)
result = optimizer.get_result_summary()
```

The current dual-wavelength loss is:

```python
loss = loss_scale * (2 - overlap_1 - overlap_2) ** 2
```

This is intentionally simple and can be replaced later if one wavelength needs different weighting.

## slmsuite Input Beam

The recommended notebooks load input amplitude and phase from slmsuite wavefront-calibration files. The notebook helper does the following:

1. Creates simulated slmsuite SLM/camera objects with the requested SLM size, pixel pitch, and wavelength.
2. Loads `wavefront_superpixel` calibration from the h5 file.
3. Runs `wavefront_calibration_superpixel_process()`.
4. Reads `amplitude` and `phase` from the calibration result.
5. Downsamples amplitude to the superpixel grid by averaging.
6. Downsamples phase using complex averaging, `angle(mean(exp(1j * phase)))`, to avoid wrapped-phase averaging errors.

The default calibration result keys are:

```python
slmsuite_amplitude_key = "amplitude"
slmsuite_phase_key = "phase"
```

If a calibration file uses different keys, the notebook raises an error listing the available keys.

## Targets, PSF, and Masks

Targets are built on the focal-plane grid. The current notebooks use a rectangular target:

```python
ideal_target = build_rectangle_target(...)
target_amplitude = apply_psf_smoothing(
    ideal_target,
    psf_sigma_x_um,
    psf_sigma_y_um,
    propagator.camera_plane.scale_um,
)
```

The PSF smoothing is applied to the target amplitude, not to the mask.

The optimization mask is passed separately to the optimizer. Common mask builders are in `loss_functions.py`:

- `build_circular_mask(...)`
- `build_rectangular_mask(...)`
- `build_threshold_expanded_mask(...)`
- `build_expanded_support_mask(...)`

The notebooks currently use a circular mask around the target.

## Superpixels and Full-Resolution Holograms

Optimization is performed on the superpixel grid for speed. The physical SLM pixel pitch and the superpixel size define the effective computational pitch:

```python
effective_pitch_um = slm_pixel_pitch_um * superpixel_size
```

After optimization, recover the full-resolution hologram with:

```python
full_resolution_hologram = optimizer.get_full_resolution_hologram()
```

This expands each optimized superpixel value back to the native SLM grid by repetition and wraps the result modulo `2*pi`.

## Plotting

For single-wavelength optimization:

```python
optimizer.plot_loss_history()
optimizer.plot_result_summary()
```

`plot_result_summary()` shows the initial hologram, final hologram, target/mask, output intensity, output phase, and mask-limited center x/y cuts.

For propagation diagnostics:

```python
propagator.plot_propagation_summary(profile_mask=None)
propagator.plot_propagation_summary(profile_mask=result.target_mask)
```

When a mask is provided, the x/y profile panel is restricted to the mask region.

For dual-wavelength optimization:

```python
optimizer.plot_result_summary()
```

This shows the shared hologram, loss history, and per-channel target/output/phase/profile summaries.

## Practical Notes

- Use `(Ny, Nx)` order for array shapes.
- SLM coordinates are indexed in array order, while focal-plane coordinates use physical micrometer sampling from the Fourier relation.
- The SLM input beam phase is separate from the hologram phase.
- `optimize_phase=True` includes target phase in the overlap objective.
- Target masks are required; optimizing without a mask is not supported by the OOP optimizer interfaces.
- `history_implementation/` contains the previous procedural implementation and old notebooks for reference only.

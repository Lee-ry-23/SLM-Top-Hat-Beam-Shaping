"""SLM Top-Hat Beam Shaping public API."""

from .array_helpers import center_crop_or_pad_array, crop_center, expand_superpixel, normalize_power
from .dual_wavelength_optimizer import DWCGOptimizer, DWChannelResult, DWOptimizationResult
from .initial_holograms import (
    axis_curvature_hologram,
    centered_pixel,
    curvature_hologram,
    lens_phase_hologram,
    random_hologram,
    zero_hologram,
)
from .loss_functions import (
    amplitude_overlap,
    build_circular_mask,
    build_expanded_support_mask,
    build_rectangular_mask,
    build_support_mask,
    build_threshold_expanded_mask,
    cg_overlap_loss,
    compute_benchmarks,
    efficiency_metric,
    phase_overlap,
)
from .optical_planes import BeamCenterFit, CameraPlane, OpticalPlane, SLMPlane
from .optimizer import CGOptimizer, OptimizationResult
from .propagation_functions import focal_plane_pixel_size_um, inverse_shifted_fourier_transform, shifted_fourier_transform
from .propagator import Propagator
from .target_profiles import apply_psf_smoothing, build_line_target, build_rectangle_target

__all__ = [
    "BeamCenterFit",
    "CameraPlane",
    "CGOptimizer",
    "DWCGOptimizer",
    "DWChannelResult",
    "DWOptimizationResult",
    "OpticalPlane",
    "OptimizationResult",
    "Propagator",
    "SLMPlane",
    "amplitude_overlap",
    "apply_psf_smoothing",
    "axis_curvature_hologram",
    "build_circular_mask",
    "build_expanded_support_mask",
    "build_line_target",
    "build_rectangle_target",
    "build_rectangular_mask",
    "build_support_mask",
    "build_threshold_expanded_mask",
    "center_crop_or_pad_array",
    "centered_pixel",
    "cg_overlap_loss",
    "compute_benchmarks",
    "crop_center",
    "curvature_hologram",
    "efficiency_metric",
    "expand_superpixel",
    "focal_plane_pixel_size_um",
    "inverse_shifted_fourier_transform",
    "lens_phase_hologram",
    "normalize_power",
    "phase_overlap",
    "random_hologram",
    "shifted_fourier_transform",
    "zero_hologram",
]

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from validation_helpers import validate_axis, validate_positive_float, validate_positive_scale_um, validate_shape


def build_rectangle_target(
    shape: tuple[int, int],
    x_axis_um: npt.ArrayLike,
    y_axis_um: npt.ArrayLike,
    width_x_um: float,
    width_y_um: float,
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    x_axis = validate_axis(x_axis_um, target_shape[1], "x_axis_um")
    y_axis = validate_axis(y_axis_um, target_shape[0], "y_axis_um")
    width_x = validate_positive_float(width_x_um, "width_x_um")
    width_y = validate_positive_float(width_y_um, "width_y_um")
    x_center = 0.5 * (float(x_axis[0]) + float(x_axis[-1]))
    y_center = 0.5 * (float(y_axis[0]) + float(y_axis[-1]))
    x_profile = _pixel_overlap_tophat_profile(x_axis, width_x, x_center, "x_axis_um")
    y_profile = _pixel_overlap_tophat_profile(y_axis, width_y, y_center, "y_axis_um")
    return np.outer(y_profile, x_profile)


def build_line_target(
    shape: tuple[int, int],
    x_axis_um: npt.ArrayLike,
    y_axis_um: npt.ArrayLike,
    width_x_um: float,
    gaussian_diameter_y_um: float,
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    x_axis = validate_axis(x_axis_um, target_shape[1], "x_axis_um")
    y_axis = validate_axis(y_axis_um, target_shape[0], "y_axis_um")
    width_x = validate_positive_float(width_x_um, "width_x_um")
    diameter_y = validate_positive_float(gaussian_diameter_y_um, "gaussian_diameter_y_um")
    x_center = 0.5 * (float(x_axis[0]) + float(x_axis[-1]))
    y_center = 0.5 * (float(y_axis[0]) + float(y_axis[-1]))
    x_profile = (np.abs(x_axis - x_center) <= width_x / 2.0).astype(float)
    y_profile = np.exp(-2.0 * ((y_axis - y_center) / (diameter_y / 2.0)) ** 2)
    return np.outer(y_profile, x_profile)


def apply_psf_smoothing(
    target: npt.ArrayLike,
    sigma_x_um: float,
    sigma_y_um: float,
    scale_um: tuple[float, float],
) -> npt.NDArray[np.float64]:
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim != 2:
        raise ValueError(f"target must be a 2D array, got shape {target_array.shape}.")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target contains NaN or infinite values.")
    sigma_x = float(sigma_x_um)
    sigma_y = float(sigma_y_um)
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError(f"PSF sigmas must be non-negative, got sigma_x_um={sigma_x_um}, sigma_y_um={sigma_y_um}.")
    dy_um, dx_um = validate_positive_scale_um(scale_um, "scale_um")
    sigma_y_px = sigma_y / dy_um if sigma_y > 0 else 0.0
    sigma_x_px = sigma_x / dx_um if sigma_x > 0 else 0.0
    smoothed = gaussian_filter(target_array, sigma=(sigma_y_px, sigma_x_px), mode="constant")
    peak = float(np.max(smoothed))
    if peak <= 0:
        raise ValueError("PSF smoothing produced an empty target.")
    return smoothed / peak


def _pixel_overlap_tophat_profile(
    axis_um: npt.NDArray[np.float64],
    width_um: float,
    center_um: float,
    axis_name: str,
) -> npt.NDArray[np.float64]:
    if axis_um.size < 2:
        raise ValueError(f"{axis_name} must contain at least two points to build a subpixel target.")
    spacing_um = float(np.median(np.diff(axis_um)))
    if spacing_um <= 0.0:
        raise ValueError(f"{axis_name} must be strictly increasing.")
    half_spacing_um = 0.5 * spacing_um
    target_min_um = float(center_um) - 0.5 * float(width_um)
    target_max_um = float(center_um) + 0.5 * float(width_um)
    pixel_min_um = axis_um - half_spacing_um
    pixel_max_um = axis_um + half_spacing_um
    overlap_um = np.maximum(0.0, np.minimum(pixel_max_um, target_max_um) - np.maximum(pixel_min_um, target_min_um))
    profile = overlap_um / spacing_um
    if float(np.max(profile)) <= 0.0:
        nearest_index = int(np.argmin(np.abs(axis_um - float(center_um))))
        profile[nearest_index] = min(1.0, float(width_um) / spacing_um)
    return profile


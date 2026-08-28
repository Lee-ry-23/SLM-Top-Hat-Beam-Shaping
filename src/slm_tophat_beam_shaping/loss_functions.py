from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .validation_helpers import (
    validate_array,
    validate_axis,
    validate_complex_2d_array,
    validate_positive_float,
    validate_positive_scale_um,
    validate_real_2d_array,
)


def amplitude_overlap(
    target: npt.ArrayLike,
    output: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> float:
    target_array = validate_real_2d_array(target, "target")
    output_array = validate_array(output, target_array.shape, "output")
    mask_array = validate_array(mask, target_array.shape, "mask")
    numerator = float(np.sum(target_array * output_array * mask_array))
    denominator = float(np.sqrt(np.sum(target_array**2) * np.sum((output_array * mask_array) ** 2)))
    if denominator <= 0:
        raise ValueError("Cannot compute amplitude overlap because the denominator is zero.")
    return numerator / denominator


def phase_overlap(
    target: npt.ArrayLike,
    output: npt.ArrayLike,
    reference_phase: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> float:
    target_array = validate_real_2d_array(target, "target")
    output_field = validate_complex_2d_array(output, "output")
    reference_array = validate_array(reference_phase, target_array.shape, "reference_phase")
    mask_array = validate_array(mask, target_array.shape, "mask")
    if output_field.shape != target_array.shape:
        raise ValueError(f"output shape must be {target_array.shape}, got {output_field.shape}.")
    output_amplitude = np.abs(output_field)
    output_phase = np.angle(output_field)
    numerator = float(np.sum(target_array * output_amplitude * mask_array * np.cos(output_phase - reference_array)))
    denominator = float(np.sqrt(np.sum(target_array**2) * np.sum((output_amplitude * mask_array) ** 2)))
    if denominator <= 0:
        raise ValueError("Cannot compute phase overlap because the denominator is zero.")
    return numerator / denominator


def efficiency_metric(output: npt.ArrayLike, mask: npt.ArrayLike) -> float:
    output_field = validate_complex_2d_array(output, "output")
    mask_array = validate_array(mask, output_field.shape, "mask")
    intensity = np.abs(output_field) ** 2
    denominator = float(np.sum(intensity))
    if denominator <= 0:
        raise ValueError("Cannot compute efficiency because output has no power.")
    return float(np.sum(intensity * mask_array) / denominator)


def cg_overlap_loss(overlap: float, loss_scale: float) -> float:
    scale = validate_positive_float(loss_scale, "loss_scale")
    return scale * (1.0 - float(overlap)) ** 2


def compute_benchmarks(
    output_field: npt.ArrayLike,
    target: npt.ArrayLike,
    reference_phase: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> dict[str, float]:
    output_array = validate_complex_2d_array(output_field, "output_field")
    target_array = validate_array(target, output_array.shape, "target")
    reference_array = validate_array(reference_phase, output_array.shape, "reference_phase")
    mask_array = validate_array(mask, output_array.shape, "mask")

    output_intensity = np.abs(output_array) ** 2
    target_intensity = target_array**2
    output_weighted = output_intensity * mask_array
    target_weighted = target_intensity * mask_array

    total_power = float(np.sum(output_intensity))
    if total_power <= 0:
        raise ValueError("Cannot compute benchmarks because output has no power.")
    efficiency = float(np.sum(output_weighted) / total_power)

    target_field = target_array * np.exp(1j * reference_array)
    target_weighted_field = target_field * mask_array
    output_weighted_field = output_array * mask_array
    overlap_numerator = float(np.abs(np.sum(np.conj(target_weighted_field) * output_weighted_field)) ** 2)
    overlap_denominator = float(np.sum(np.abs(target_weighted_field) ** 2) * np.sum(np.abs(output_weighted_field) ** 2))
    fidelity = overlap_numerator / max(overlap_denominator, np.finfo(float).eps)

    active = mask_array > 0
    if not np.any(active):
        raise ValueError("Cannot compute benchmarks because mask has no active pixels.")
    output_norm = np.zeros_like(output_intensity)
    target_norm = np.zeros_like(target_intensity)
    output_norm[active] = output_weighted[active] / max(float(np.sum(output_weighted[active])), np.finfo(float).eps)
    target_norm[active] = target_weighted[active] / max(float(np.sum(target_weighted[active])), np.finfo(float).eps)
    valid = active & (target_norm > 1e-6)
    if not np.any(valid):
        rms_error = np.nan
    else:
        relative_error = ((output_norm[valid] - target_norm[valid]) / target_norm[valid]) ** 2
        rms_error = float(np.sqrt(np.mean(relative_error)))

    phase_difference = np.angle(np.exp(1j * (np.angle(output_array) - reference_array)))
    phase_error = float(np.sqrt(np.sum((phase_difference**2) * mask_array) / max(float(np.sum(mask_array)), np.finfo(float).eps)))

    return {
        "efficiency": efficiency,
        "fidelity": float(fidelity),
        "rms_error": rms_error,
        "phase_error": phase_error,
    }


def build_support_mask(target: npt.ArrayLike, threshold_fraction: float) -> npt.NDArray[np.float64]:
    target_array = validate_real_2d_array(target, "target")
    threshold = _validate_nonnegative_float(threshold_fraction, "threshold_fraction")
    peak = float(np.max(np.abs(target_array)))
    if peak <= 0:
        raise ValueError("Cannot build support mask because target has no positive values.")
    return (np.abs(target_array) > threshold * peak).astype(float)


def build_circular_mask(
    shape: tuple[int, int],
    x_axis_um: npt.ArrayLike,
    y_axis_um: npt.ArrayLike,
    center_x_um: float,
    center_y_um: float,
    radius_um: float,
) -> npt.NDArray[np.float64]:
    y_size, x_size = shape
    x_axis = validate_axis(x_axis_um, x_size, "x_axis_um")
    y_axis = validate_axis(y_axis_um, y_size, "y_axis_um")
    radius = validate_positive_float(radius_um, "radius_um")
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    distance_squared = (x_grid - float(center_x_um)) ** 2 + (y_grid - float(center_y_um)) ** 2
    return (distance_squared <= radius**2).astype(float)


def build_rectangular_mask(
    shape: tuple[int, int],
    x_axis_um: npt.ArrayLike,
    y_axis_um: npt.ArrayLike,
    center_x_um: float,
    center_y_um: float,
    width_x_um: float,
    width_y_um: float,
) -> npt.NDArray[np.float64]:
    y_size, x_size = shape
    x_axis = validate_axis(x_axis_um, x_size, "x_axis_um")
    y_axis = validate_axis(y_axis_um, y_size, "y_axis_um")
    half_width_x = validate_positive_float(width_x_um, "width_x_um") / 2.0
    half_width_y = validate_positive_float(width_y_um, "width_y_um") / 2.0
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    active = (np.abs(x_grid - float(center_x_um)) <= half_width_x) & (np.abs(y_grid - float(center_y_um)) <= half_width_y)
    return active.astype(float)


def build_threshold_expanded_mask(
    intensity: npt.ArrayLike,
    threshold: float,
    expansion_um: float,
    scale_um: tuple[float, float],
) -> npt.NDArray[np.float64]:
    intensity_array = validate_real_2d_array(intensity, "intensity")
    threshold_value = _validate_nonnegative_float(threshold, "threshold")
    expansion = _validate_nonnegative_float(expansion_um, "expansion_um")
    scale = validate_positive_scale_um(scale_um, "scale_um")
    support = intensity_array > threshold_value
    if not np.any(support):
        raise ValueError("Cannot build threshold-expanded mask because the threshold removes every pixel.")
    if expansion == 0.0:
        return support.astype(float)
    from scipy.ndimage import distance_transform_edt

    distance_to_support_um = distance_transform_edt(~support, sampling=scale)
    return (distance_to_support_um <= expansion).astype(float)


def build_expanded_support_mask(
    target: npt.ArrayLike,
    threshold_fraction: float,
    expansion_um: float,
    scale_um: tuple[float, float],
) -> npt.NDArray[np.float64]:
    target_array = validate_real_2d_array(target, "target")
    threshold = _validate_nonnegative_float(threshold_fraction, "threshold_fraction")
    peak = float(np.max(np.abs(target_array)))
    if peak <= 0:
        raise ValueError("Cannot build expanded support mask because target has no positive values.")
    return build_threshold_expanded_mask(target_array, threshold * peak, expansion_um, scale_um)


def _validate_nonnegative_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite float, got {value}.")
    return result



from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .validation_helpers import validate_positive_scale_um, validate_shape


def random_hologram(shape: tuple[int, int], seed: int | None) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=target_shape)


def zero_hologram(shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    return np.zeros(target_shape, dtype=float)


def curvature_hologram(
    shape: tuple[int, int],
    linear_tilt: float,
    astigmatism_weight: float,
    quadratic_curvature: float,
    linear_angle_rad: float,
    conical_weight: float,
    center: tuple[float, float],
) -> npt.NDArray[np.float64]:
    curvature_x = 3.0 * float(quadratic_curvature) * float(astigmatism_weight)
    curvature_y = 3.0 * float(quadratic_curvature) * (1.0 - float(astigmatism_weight))
    return axis_curvature_hologram(
        shape,
        linear_tilt,
        curvature_x,
        curvature_y,
        linear_angle_rad,
        conical_weight,
        center,
    )


def axis_curvature_hologram(
    shape: tuple[int, int],
    linear_tilt: float,
    quadratic_curvature_x: float,
    quadratic_curvature_y: float,
    linear_angle_rad: float,
    conical_weight: float,
    center: tuple[float, float],
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    center_y, center_x = _validate_center_pixel(center, target_shape, "center")
    ny, nx = target_shape
    x = np.arange(nx, dtype=float) - center_x
    y = np.arange(ny, dtype=float) - center_y
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    angle = float(linear_angle_rad)
    linear_phase = float(linear_tilt) * (x_grid * np.cos(angle) + y_grid * np.sin(angle))
    quadratic_phase = float(quadratic_curvature_x) * x_grid**2 + float(quadratic_curvature_y) * y_grid**2
    conical_phase = float(conical_weight) * np.sqrt(x_grid**2 + y_grid**2)
    phase = conical_phase + quadratic_phase + linear_phase
    return np.mod(phase, 2.0 * np.pi)


def lens_phase_hologram(
    shape: tuple[int, int],
    scale_um: tuple[float, float],
    curvature: float,
    astigmatism: float,
    angle_rad: float,
    offset: float,
    center: tuple[float, float],
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    center_y, center_x = _validate_center_pixel(center, target_shape, "center")
    dy_um, dx_um = validate_positive_scale_um(scale_um, "scale_um")
    y_um = (np.arange(target_shape[0], dtype=float) - center_y) * dy_um
    x_um = (np.arange(target_shape[1], dtype=float) - center_x) * dx_um
    x_grid, y_grid = np.meshgrid(x_um, y_um, indexing="xy")
    cos_t = np.cos(float(angle_rad))
    sin_t = np.sin(float(angle_rad))
    x_rot = cos_t * x_grid + sin_t * y_grid
    y_rot = -sin_t * x_grid + cos_t * y_grid
    phase = float(curvature) * (x_rot**2 + y_rot**2) + float(astigmatism) * (x_rot**2 - y_rot**2) + float(offset)
    return np.mod(phase, 2.0 * np.pi)


def centered_pixel(shape: tuple[int, int]) -> tuple[float, float]:
    target_shape = validate_shape(shape, "shape")
    return target_shape[0] / 2.0, target_shape[1] / 2.0


def _validate_center_pixel(center: tuple[float, float], shape: tuple[int, int], name: str) -> tuple[float, float]:
    if len(center) != 2:
        raise ValueError(f"{name} must have length 2 as (center_y, center_x), got {center}.")
    center_y = float(center[0])
    center_x = float(center[1])
    if not np.isfinite(center_y) or not np.isfinite(center_x):
        raise ValueError(f"{name} entries must be finite, got {center}.")
    if center_y < 0.0 or center_y > shape[0] - 1 or center_x < 0.0 or center_x > shape[1] - 1:
        raise ValueError(f"{name} must lie inside shape {shape}, got {center}.")
    return center_y, center_x

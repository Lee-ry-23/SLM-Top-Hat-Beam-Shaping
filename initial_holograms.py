from __future__ import annotations

import numpy as np
import numpy.typing as npt

from validation_helpers import validate_positive_float, validate_positive_scale_um, validate_shape


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
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    ny, nx = target_shape
    x = np.arange(nx, dtype=float) - nx / 2.0
    y = np.arange(ny, dtype=float) - ny / 2.0
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    angle = float(linear_angle_rad)
    linear_phase = float(linear_tilt) * (x_grid * np.cos(angle) + y_grid * np.sin(angle))
    quadratic_phase = 3.0 * float(quadratic_curvature) * (float(astigmatism_weight) * x_grid**2 + (1.0 - float(astigmatism_weight)) * y_grid**2)
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
) -> npt.NDArray[np.float64]:
    target_shape = validate_shape(shape, "shape")
    dy_um, dx_um = validate_positive_scale_um(scale_um, "scale_um")
    y_um = (np.arange(target_shape[0], dtype=float) - target_shape[0] / 2.0) * dy_um
    x_um = (np.arange(target_shape[1], dtype=float) - target_shape[1] / 2.0) * dx_um
    x_grid, y_grid = np.meshgrid(x_um, y_um, indexing="xy")
    cos_t = np.cos(float(angle_rad))
    sin_t = np.sin(float(angle_rad))
    x_rot = cos_t * x_grid + sin_t * y_grid
    y_rot = -sin_t * x_grid + cos_t * y_grid
    phase = float(curvature) * (x_rot**2 + y_rot**2) + float(astigmatism) * (x_rot**2 - y_rot**2) + float(offset)
    return np.mod(phase, 2.0 * np.pi)

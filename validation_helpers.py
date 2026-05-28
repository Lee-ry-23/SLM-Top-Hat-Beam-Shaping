from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt


UnitType = Literal["xy", "kxy"]
T = TypeVar("T")


def validate_shape(shape: tuple[int, int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must have length 2 as (Ny, Nx), got {shape}.")
    ny, nx = int(shape[0]), int(shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError(f"{name} entries must be positive, got {shape}.")
    return ny, nx


def validate_unit_type(unit_type: str) -> UnitType:
    if unit_type not in ("xy", "kxy"):
        raise ValueError(f"unit_type must be 'xy' or 'kxy', got {unit_type}.")
    return cast(UnitType, unit_type)


def validate_choice(value: str, choices: Sequence[str], name: str) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value}.")
    return value


def validate_positive_float(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite float, got {value}.")
    return result


def validate_optional_positive_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return validate_positive_float(value, name)


def validate_positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return result


def validate_scale_um(scale_um: tuple[float, float], name: str) -> tuple[float, float]:
    if len(scale_um) != 2:
        raise ValueError(f"{name} must have length 2 as (dy_um, dx_um), got {scale_um}.")
    dy_um = float(scale_um[0])
    dx_um = float(scale_um[1])
    if dy_um < 0 or dx_um < 0:
        raise ValueError(f"{name} entries must be non-negative, got {scale_um}.")
    return dy_um, dx_um


def validate_positive_scale_um(scale_um: tuple[float, float], name: str) -> tuple[float, float]:
    if len(scale_um) != 2:
        raise ValueError(f"{name} must have length 2 as (y_um, x_um), got {scale_um}.")
    y_um = validate_positive_float(scale_um[0], f"{name}[0]")
    x_um = validate_positive_float(scale_um[1], f"{name}[1]")
    return y_um, x_um


def validate_array(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def validate_amplitude(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.float64]:
    array = validate_array(data, shape, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def validate_intensity(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.float64]:
    array = validate_array(data, shape, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def validate_phase(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.float64]:
    return validate_array(data, shape, name)


def validate_field(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.complex128]:
    array = np.asarray(data, dtype=np.complex128)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def validate_complex_2d_array(data: npt.ArrayLike, name: str) -> npt.NDArray[np.complex128]:
    array = np.asarray(data, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def validate_instance(value: object, expected_type: type[T], name: str) -> T:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}, got {type(value).__name__}.")
    return value


def validate_axis(axis: npt.ArrayLike, expected_length: int, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(axis, dtype=float)
    if array.ndim != 1 or array.size != expected_length:
        raise ValueError(f"{name} must be a 1D array with length {expected_length}, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def validate_real_2d_array(data: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    shape = validate_shape((array.shape[0], array.shape[1]), f"{name}.shape")
    return validate_array(array, shape, name)


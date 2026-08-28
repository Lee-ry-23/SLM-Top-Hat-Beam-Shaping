from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .validation_helpers import validate_amplitude, validate_array, validate_positive_float, validate_positive_int, validate_shape


def ensure_2d_float_array(data: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def validate_array_shape(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> None:
    array = np.asarray(data)
    expected_shape = validate_shape(shape, "shape")
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}, got {array.shape}.")


def normalize_power(amplitude: npt.ArrayLike, target_power: float) -> npt.NDArray[np.float64]:
    amplitude_array = ensure_2d_float_array(amplitude, "amplitude")
    amplitude_array = validate_amplitude(amplitude_array, amplitude_array.shape, "amplitude")
    power = float(np.sum(amplitude_array**2))
    desired_power = validate_positive_float(target_power, "target_power")
    if power <= 0:
        raise ValueError("amplitude contains no positive power and cannot be normalized.")
    return amplitude_array * np.sqrt(desired_power / power)


def crop_center(data: npt.ArrayLike, output_shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    array = ensure_2d_float_array(data, "data")
    target_shape = validate_shape(output_shape, "output_shape")
    if target_shape[0] > array.shape[0] or target_shape[1] > array.shape[1]:
        raise ValueError(
            "output_shape must be less than or equal to data shape for center cropping. "
            f"data_shape={array.shape}, output_shape={target_shape}."
        )
    y0 = (array.shape[0] - target_shape[0]) // 2
    x0 = (array.shape[1] - target_shape[1]) // 2
    return array[y0:y0 + target_shape[0], x0:x0 + target_shape[1]].copy()


def center_crop_or_pad_array(data: npt.ArrayLike, output_shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    array = ensure_2d_float_array(data, "data")
    target_shape = validate_shape(output_shape, "output_shape")
    crop_shape = (min(array.shape[0], target_shape[0]), min(array.shape[1], target_shape[1]))
    cropped = crop_center(array, crop_shape)
    output = np.zeros(target_shape, dtype=float)
    y0 = (target_shape[0] - cropped.shape[0]) // 2
    x0 = (target_shape[1] - cropped.shape[1]) // 2
    output[y0:y0 + cropped.shape[0], x0:x0 + cropped.shape[1]] = cropped
    return output


def expand_superpixel(data: npt.ArrayLike, superpixel_size: int) -> npt.NDArray[np.float64]:
    array = ensure_2d_float_array(data, "data")
    factor = validate_positive_int(superpixel_size, "superpixel_size")
    expanded_y = np.repeat(array, factor, axis=0)
    return np.repeat(expanded_y, factor, axis=1)


def validate_real_array(data: npt.ArrayLike, shape: tuple[int, int], name: str) -> npt.NDArray[np.float64]:
    expected_shape = validate_shape(shape, "shape")
    return validate_array(data, expected_shape, name)

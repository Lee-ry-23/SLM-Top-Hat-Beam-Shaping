from __future__ import annotations

import numpy as np
import numpy.typing as npt

from validation_helpers import validate_complex_2d_array, validate_positive_float, validate_positive_scale_um, validate_shape


def shifted_fourier_transform(field: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    field_array = validate_complex_2d_array(field, "field")
    transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_array)))
    return np.asarray(transformed, dtype=np.complex128)


def inverse_shifted_fourier_transform(field: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    field_array = validate_complex_2d_array(field, "field")
    transformed = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field_array)))
    return np.asarray(transformed, dtype=np.complex128)


def pad_field_to_shape(
    field: npt.ArrayLike,
    output_shape: tuple[int, int],
) -> npt.NDArray[np.complex128]:
    field_array = validate_complex_2d_array(field, "field")
    target_shape = validate_shape(output_shape, "output_shape")
    if target_shape[0] < field_array.shape[0] or target_shape[1] < field_array.shape[1]:
        raise ValueError(
            "output_shape must be greater than or equal to field shape for padding. "
            f"field_shape={field_array.shape}, output_shape={target_shape}."
        )

    output = np.zeros(target_shape, dtype=np.complex128)
    y0 = (target_shape[0] - field_array.shape[0]) // 2
    x0 = (target_shape[1] - field_array.shape[1]) // 2
    output[y0:y0 + field_array.shape[0], x0:x0 + field_array.shape[1]] = field_array
    return output


def center_crop_or_pad_field(
    field: npt.ArrayLike,
    output_shape: tuple[int, int],
) -> npt.NDArray[np.complex128]:
    field_array = validate_complex_2d_array(field, "field")
    target_shape = validate_shape(output_shape, "output_shape")
    cropped = _center_crop_field(field_array, (min(field_array.shape[0], target_shape[0]), min(field_array.shape[1], target_shape[1])))
    output = np.zeros(target_shape, dtype=np.complex128)
    y0 = (target_shape[0] - cropped.shape[0]) // 2
    x0 = (target_shape[1] - cropped.shape[1]) // 2
    output[y0:y0 + cropped.shape[0], x0:x0 + cropped.shape[1]] = cropped
    return output


def focal_plane_pixel_size_um(
    wavelength_nm: float,
    focal_length_mm: float,
    slm_size_um: tuple[float, float],
) -> tuple[float, float]:
    wavelength_um = validate_positive_float(wavelength_nm, "wavelength_nm") / 1000.0
    focal_length_um = validate_positive_float(focal_length_mm, "focal_length_mm") * 1000.0
    slm_size_y_um, slm_size_x_um = validate_positive_scale_um(slm_size_um, "slm_size_um")
    delta_x_um = wavelength_um * focal_length_um / slm_size_x_um
    delta_y_um = wavelength_um * focal_length_um / slm_size_y_um
    return delta_y_um, delta_x_um


def _center_crop_field(
    field: npt.NDArray[np.complex128],
    output_shape: tuple[int, int],
) -> npt.NDArray[np.complex128]:
    target_shape = validate_shape(output_shape, "output_shape")
    if target_shape[0] > field.shape[0] or target_shape[1] > field.shape[1]:
        raise ValueError(
            "output_shape must be less than or equal to field shape for cropping. "
            f"field_shape={field.shape}, output_shape={target_shape}."
        )
    y0 = (field.shape[0] - target_shape[0]) // 2
    x0 = (field.shape[1] - target_shape[1]) // 2
    return field[y0:y0 + target_shape[0], x0:x0 + target_shape[1]].copy()


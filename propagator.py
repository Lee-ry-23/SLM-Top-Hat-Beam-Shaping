from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optical_planes import CameraPlane, OpticalPlane, SLMPlane
from propagation_functions import center_crop_or_pad_field, focal_plane_pixel_size_um, pad_field_to_shape, shifted_fourier_transform
from array_helpers import center_crop_or_pad_array
from validation_helpers import validate_choice, validate_instance, validate_positive_float, validate_positive_int, validate_real_2d_array


class Propagator:
    def __init__(
        self,
        slm_plane: SLMPlane,
        camera_plane: CameraPlane,
        focal_length_mm: float,
        padding_factor: int,
    ) -> None:
        self.slm_plane = validate_instance(slm_plane, SLMPlane, "slm_plane")
        self.camera_plane = validate_instance(camera_plane, CameraPlane, "camera_plane")
        self.focal_length_mm = validate_positive_float(focal_length_mm, "focal_length_mm")
        self.padding_factor = validate_positive_int(padding_factor, "padding_factor")
        self.padding_farfield = self._create_zero_padding_farfield((0.0, 0.0))
        self.update_camera_sampling()

    def set_slm_plane(self, slm_plane: SLMPlane) -> None:
        self.slm_plane = validate_instance(slm_plane, SLMPlane, "slm_plane")
        self.update_padding_farfield()
        self.update_camera_sampling()

    def set_camera_plane(self, camera_plane: CameraPlane) -> None:
        self.camera_plane = validate_instance(camera_plane, CameraPlane, "camera_plane")
        self.update_camera_sampling()

    def set_focal_length(self, focal_length_mm: float) -> None:
        self.focal_length_mm = validate_positive_float(focal_length_mm, "focal_length_mm")
        self.update_camera_sampling()

    def set_padding_factor(self, padding_factor: int) -> None:
        self.padding_factor = validate_positive_int(padding_factor, "padding_factor")
        self.update_padding_farfield()
        self.update_camera_sampling()

    def propagate(self, method: str) -> None:
        propagation_method = validate_choice(method, ("sft",), "method")
        if propagation_method == "sft":
            self.propagate_sft()
            return
        raise ValueError(f"Unsupported propagation method {method}.")

    def propagate_sft(self) -> None:
        self.update_padding_farfield()
        self.update_camera_sampling()
        padded_slm_field = pad_field_to_shape(self.slm_plane.get_modulated_field(), self.padding_farfield.shape)
        farfield = shifted_fourier_transform(padded_slm_field)
        self.padding_farfield.set_field(farfield)
        camera_field = center_crop_or_pad_field(farfield, self.camera_plane.shape)
        self.camera_plane.set_field(camera_field)

    def update_padding_farfield(self) -> None:
        expected_shape = self._expected_padding_shape()
        if self.padding_farfield.shape == expected_shape:
            self.padding_farfield.wavelength_nm = self.slm_plane.wavelength_nm
            self.padding_farfield.unit_type = "xy"
            return

        self.padding_farfield = self._create_zero_padding_farfield(self.padding_farfield.scale_um)

    def update_camera_sampling(self) -> None:
        sampling_um = focal_plane_pixel_size_um(
            self.slm_plane.wavelength_nm,
            self.focal_length_mm,
            self._slm_size_um(),
        )
        self.padding_farfield.scale_um = sampling_um
        self.camera_plane.scale_um = sampling_um
        self.camera_plane.wavelength_nm = self.slm_plane.wavelength_nm

    def get_slm_field(self) -> npt.NDArray[np.complex128]:
        return self.slm_plane.get_modulated_field()

    def get_padding_farfield(self) -> OpticalPlane:
        return self.padding_farfield.copy()

    def get_camera_field(self) -> npt.NDArray[np.complex128]:
        return self.camera_plane.get_field()

    def plot_propagation_summary(self, profile_mask: npt.ArrayLike | None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)
        _draw_image(axes[0, 0], self.slm_plane.intensity, self.slm_plane.x_axis_um, self.slm_plane.y_axis_um, "SLM input intensity", "magma")
        _draw_image(axes[0, 1], self.slm_plane.hologram, self.slm_plane.x_axis_um, self.slm_plane.y_axis_um, "SLM hologram", "twilight")
        _draw_image(axes[0, 2], self.camera_plane.intensity, self.camera_plane.x_axis_um, self.camera_plane.y_axis_um, "Camera-plane intensity", "inferno")
        _draw_image(
            axes[1, 0],
            self.padding_farfield.intensity,
            self.padding_farfield.x_axis_um,
            self.padding_farfield.y_axis_um,
            "Padded far-field intensity",
            "inferno",
        )
        _draw_image(axes[1, 1], self.camera_plane.phase, self.camera_plane.x_axis_um, self.camera_plane.y_axis_um, "Camera-plane phase", "twilight")
        _draw_center_profiles(
            axes[1, 2],
            self.camera_plane.intensity,
            self.camera_plane.x_axis_um,
            self.camera_plane.y_axis_um,
            _profile_mask_for_shape(profile_mask, self.camera_plane.shape),
            "Camera-plane center intensity cuts",
        )
        return fig

    def _expected_padding_shape(self) -> tuple[int, int]:
        return (
            self.slm_plane.shape[0] * self.padding_factor,
            self.slm_plane.shape[1] * self.padding_factor,
        )

    def _create_zero_padding_farfield(self, scale_um: tuple[float, float]) -> OpticalPlane:
        expected_shape = self._expected_padding_shape()
        zero_amplitude = np.zeros(expected_shape, dtype=float)
        zero_phase = np.zeros(expected_shape, dtype=float)
        return OpticalPlane(
            expected_shape,
            self.slm_plane.wavelength_nm,
            "xy",
            scale_um,
            zero_amplitude,
            zero_phase,
        )

    def _slm_size_um(self) -> tuple[float, float]:
        return (
            self.slm_plane.shape[0] * self.slm_plane.scale_um[0],
            self.slm_plane.shape[1] * self.slm_plane.scale_um[1],
        )


def _draw_image(
    ax: plt.Axes,
    data: npt.NDArray[np.float64],
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
    title: str,
    cmap: str,
) -> None:
    image = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        extent=_axis_extent_um(x_axis_um, y_axis_um),
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.figure.colorbar(image, ax=ax, shrink=0.82)


def _draw_center_profiles(
    ax: plt.Axes,
    intensity: npt.NDArray[np.float64],
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
    profile_mask: npt.NDArray[np.float64] | None,
    title: str,
) -> None:
    center_y = intensity.shape[0] // 2
    center_x = intensity.shape[1] // 2
    x_profile = intensity[center_y, :]
    y_profile = intensity[:, center_x]
    x_axis = x_axis_um
    y_axis = y_axis_um
    if profile_mask is not None:
        x_active = profile_mask[center_y, :] > 0
        y_active = profile_mask[:, center_x] > 0
        if not np.any(x_active):
            raise ValueError("profile_mask has no active pixels on the center x cut.")
        if not np.any(y_active):
            raise ValueError("profile_mask has no active pixels on the center y cut.")
        x_profile = x_profile[x_active]
        y_profile = y_profile[y_active]
        x_axis = x_axis_um[x_active]
        y_axis = y_axis_um[y_active]
    if np.max(x_profile) > 0:
        x_profile = x_profile / np.max(x_profile)
    if np.max(y_profile) > 0:
        y_profile = y_profile / np.max(y_profile)
    ax.plot(x_axis, x_profile, lw=2.0, label="x cut")
    ax.plot(y_axis, y_profile, lw=2.0, ls="--", label="y cut")
    ax.set_title(title)
    ax.set_xlabel("position (um)")
    ax.set_ylabel("normalized intensity")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8)


def _profile_mask_for_shape(
    profile_mask: npt.ArrayLike | None,
    shape: tuple[int, int],
) -> npt.NDArray[np.float64] | None:
    if profile_mask is None:
        return None
    mask = validate_real_2d_array(profile_mask, "profile_mask")
    if mask.shape != shape:
        mask = center_crop_or_pad_array(mask, shape)
    if not np.any(mask > 0):
        raise ValueError("profile_mask has no active pixels after mapping to the camera-plane shape.")
    return mask


def _axis_extent_um(
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    return (
        float(x_axis_um[0]),
        float(_axis_last_value_um(x_axis_um)),
        float(y_axis_um[0]),
        float(_axis_last_value_um(y_axis_um)),
    )


def _axis_last_value_um(axis_um: npt.NDArray[np.float64]) -> float:
    if axis_um.size <= 1 or float(axis_um[-1]) <= float(axis_um[0]):
        return float(axis_um[0] + max(axis_um.size - 1, 1))
    return float(axis_um[-1])

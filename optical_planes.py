from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from validation_helpers import (
    UnitType,
    validate_amplitude,
    validate_field,
    validate_intensity,
    validate_optional_positive_float,
    validate_phase,
    validate_positive_float,
    validate_positive_int,
    validate_scale_um,
    validate_shape,
    validate_unit_type,
)


class OpticalPlane:
    def __init__(
        self,
        shape: tuple[int, int],
        wavelength_nm: float,
        unit_type: UnitType,
        scale_um: tuple[float, float],
        amplitude: npt.ArrayLike,
        phase: npt.ArrayLike,
    ) -> None:
        self.shape = validate_shape(shape, "shape")
        self.wavelength_nm = validate_positive_float(wavelength_nm, "wavelength_nm")
        self.unit_type = validate_unit_type(unit_type)
        self.scale_um = validate_scale_um(scale_um, "scale_um")
        self.amplitude = validate_amplitude(amplitude, self.shape, "amplitude")
        self.phase = validate_phase(phase, self.shape, "phase")

    @property
    def field(self) -> npt.NDArray[np.complex128]:
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def intensity(self) -> npt.NDArray[np.float64]:
        return self.amplitude**2

    @property
    def x_axis_um(self) -> npt.NDArray[np.float64]:
        return np.arange(self.shape[1], dtype=float) * self.scale_um[1]

    @property
    def y_axis_um(self) -> npt.NDArray[np.float64]:
        return np.arange(self.shape[0], dtype=float) * self.scale_um[0]

    def set_amplitude(self, amplitude: npt.ArrayLike) -> None:
        self.amplitude = validate_amplitude(amplitude, self.shape, "amplitude")

    def set_phase(self, phase: npt.ArrayLike) -> None:
        self.phase = validate_phase(phase, self.shape, "phase")

    def set_field(self, field: npt.ArrayLike) -> None:
        field_array = validate_field(field, self.shape, "field")
        self.amplitude = np.abs(field_array)
        self.phase = np.angle(field_array)

    def set_intensity(self, intensity: npt.ArrayLike) -> None:
        intensity_array = validate_intensity(intensity, self.shape, "intensity")
        self.amplitude = np.sqrt(intensity_array)

    def get_field(self) -> npt.NDArray[np.complex128]:
        return self.field.copy()

    def get_intensity(self) -> npt.NDArray[np.float64]:
        return self.intensity.copy()

    def get_phase(self) -> npt.NDArray[np.float64]:
        return self.phase.copy()

    def copy(self) -> "OpticalPlane":
        return OpticalPlane(
            self.shape,
            self.wavelength_nm,
            self.unit_type,
            self.scale_um,
            self.amplitude.copy(),
            self.phase.copy(),
        )

    def plot_intensity(self) -> plt.Figure:
        return _plot_plane_image(
            self.intensity,
            self.x_axis_um,
            self.y_axis_um,
            "Intensity",
            "intensity",
            "magma",
        )

    def plot_phase(self) -> plt.Figure:
        return _plot_plane_image(
            self.phase,
            self.x_axis_um,
            self.y_axis_um,
            "Phase",
            "phase (rad)",
            "twilight",
        )

    def plot_field_summary(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        _draw_plane_image(axes[0], self.intensity, self.x_axis_um, self.y_axis_um, "Intensity", "intensity", "magma")
        _draw_plane_image(axes[1], self.phase, self.x_axis_um, self.y_axis_um, "Phase", "phase (rad)", "twilight")
        return fig


class SLMPlane(OpticalPlane):
    def __init__(
        self,
        shape: tuple[int, int],
        wavelength_nm: float,
        pixel_pitch_um: float,
        superpixel_size: int,
        amplitude: npt.ArrayLike,
        phase: npt.ArrayLike,
        hologram: npt.ArrayLike,
    ) -> None:
        self.pixel_pitch_um = validate_positive_float(pixel_pitch_um, "pixel_pitch_um")
        self.superpixel_size = validate_positive_int(superpixel_size, "superpixel_size")
        self.hologram = np.mod(validate_phase(hologram, validate_shape(shape, "shape"), "hologram"), 2 * np.pi)
        effective_pitch_um = self.pixel_pitch_um * self.superpixel_size
        super().__init__(
            shape,
            wavelength_nm,
            "xy",
            (effective_pitch_um, effective_pitch_um),
            amplitude,
            phase,
        )

    @property
    def modulated_phase(self) -> npt.NDArray[np.float64]:
        return np.mod(self.phase + self.hologram, 2 * np.pi)

    @property
    def modulated_field(self) -> npt.NDArray[np.complex128]:
        return self.amplitude * np.exp(1j * self.modulated_phase)

    def set_hologram(self, hologram: npt.ArrayLike) -> None:
        self.hologram = np.mod(validate_phase(hologram, self.shape, "hologram"), 2 * np.pi)

    def get_hologram(self) -> npt.NDArray[np.float64]:
        return self.hologram.copy()

    def get_modulated_field(self) -> npt.NDArray[np.complex128]:
        return self.modulated_field.copy()

    def get_modulated_phase(self) -> npt.NDArray[np.float64]:
        return self.modulated_phase.copy()

    def get_full_resolution_hologram(self) -> npt.NDArray[np.float64]:
        expanded_y = np.repeat(self.hologram, self.superpixel_size, axis=0)
        expanded_xy = np.repeat(expanded_y, self.superpixel_size, axis=1)
        return np.mod(expanded_xy, 2 * np.pi)

    def plot_hologram(self) -> plt.Figure:
        return _plot_plane_image(
            self.hologram,
            self.x_axis_um,
            self.y_axis_um,
            "SLM hologram",
            "phase (rad)",
            "twilight",
        )

    def plot_input_beam(self) -> plt.Figure:
        return _plot_plane_image(
            self.intensity,
            self.x_axis_um,
            self.y_axis_um,
            "SLM input beam intensity",
            "intensity",
            "magma",
        )


class CameraPlane(OpticalPlane):
    def __init__(
        self,
        shape: tuple[int, int],
        wavelength_nm: float,
        scale_um: tuple[float, float],
        camera_pixel_pitch_um: float | None,
        amplitude: npt.ArrayLike,
        phase: npt.ArrayLike,
    ) -> None:
        self.camera_pixel_pitch_um = validate_optional_positive_float(camera_pixel_pitch_um, "camera_pixel_pitch_um")
        super().__init__(shape, wavelength_nm, "xy", scale_um, amplitude, phase)

    def set_camera_pixel_pitch(self, pixel_pitch_um: float) -> None:
        self.camera_pixel_pitch_um = validate_positive_float(pixel_pitch_um, "pixel_pitch_um")

    def set_measured_intensity(self, intensity: npt.ArrayLike) -> None:
        self.set_intensity(intensity)

    def set_measured_amplitude(self, amplitude: npt.ArrayLike) -> None:
        self.set_amplitude(amplitude)


def _plot_plane_image(
    data: npt.NDArray[np.float64],
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
    title: str,
    colorbar_label: str,
    cmap: str,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.8), constrained_layout=True)
    _draw_plane_image(ax, data, x_axis_um, y_axis_um, title, colorbar_label, cmap)
    return fig


def _draw_plane_image(
    ax: plt.Axes,
    data: npt.NDArray[np.float64],
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
    title: str,
    colorbar_label: str,
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
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label(colorbar_label)


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

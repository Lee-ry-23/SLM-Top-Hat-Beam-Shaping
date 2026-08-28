from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize

from .validation_helpers import (
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


@dataclass(frozen=True)
class BeamCenterFit:
    center_pixel: tuple[float, float]
    center_um: tuple[float, float]
    sigma_pixel: tuple[float, float]
    sigma_um: tuple[float, float]
    background: float
    peak_amplitude: float


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

    def fit_amplitude_center(self, threshold_fraction: float, maxfev: int) -> BeamCenterFit:
        return _fit_gaussian_amplitude_center(
            self.amplitude,
            self.scale_um,
            _validate_open_unit_interval(threshold_fraction, "threshold_fraction"),
            validate_positive_int(maxfev, "maxfev"),
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


def _fit_gaussian_amplitude_center(
    amplitude: npt.NDArray[np.float64],
    scale_um: tuple[float, float],
    threshold_fraction: float,
    maxfev: int,
) -> BeamCenterFit:
    min_amplitude = float(np.min(amplitude))
    max_amplitude = float(np.max(amplitude))
    dynamic_range = max_amplitude - min_amplitude
    if dynamic_range <= 0.0:
        raise ValueError("SLM amplitude has no positive dynamic range for center fitting.")

    normalized = (amplitude - min_amplitude) / dynamic_range
    support = normalized >= threshold_fraction
    if not np.any(support):
        raise ValueError(f"No amplitude pixels exceed threshold_fraction={threshold_fraction}.")

    support_y, support_x = np.where(support)
    y0, y1 = _expanded_bounds(int(support_y.min()), int(support_y.max()), amplitude.shape[0])
    x0, x1 = _expanded_bounds(int(support_x.min()), int(support_x.max()), amplitude.shape[1])

    roi = amplitude[y0:y1, x0:x1]
    y = np.arange(y0, y1, dtype=float)
    x = np.arange(x0, x1, dtype=float)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    initial_background = float(np.percentile(roi, 5.0))
    signal = np.clip(roi - initial_background, 0.0, None)
    signal_sum = float(np.sum(signal))
    if signal_sum <= 0.0:
        raise ValueError("Amplitude ROI has no positive signal above background for center fitting.")

    initial_center_y = float(np.sum(y_grid * signal) / signal_sum)
    initial_center_x = float(np.sum(x_grid * signal) / signal_sum)
    initial_sigma_y = max(float(np.sqrt(np.sum((y_grid - initial_center_y) ** 2 * signal) / signal_sum)), 0.5)
    initial_sigma_x = max(float(np.sqrt(np.sum((x_grid - initial_center_x) ** 2 * signal) / signal_sum)), 0.5)
    initial_peak = max(max_amplitude - initial_background, np.finfo(float).eps)

    lower_bounds = [min_amplitude - dynamic_range, 0.0, float(y0), float(x0), 0.5, 0.5]
    upper_bounds = [max_amplitude, 2.0 * dynamic_range, float(y1 - 1), float(x1 - 1), float(amplitude.shape[0]), float(amplitude.shape[1])]
    p0 = [initial_background, initial_peak, initial_center_y, initial_center_x, initial_sigma_y, initial_sigma_x]

    fitted, _ = scipy.optimize.curve_fit(
        _elliptical_gaussian_amplitude,
        (y_grid.reshape(-1), x_grid.reshape(-1)),
        roi.reshape(-1),
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=maxfev,
    )

    background, peak_amplitude, center_y, center_x, sigma_y, sigma_x = (float(value) for value in fitted)
    dy_um, dx_um = scale_um
    return BeamCenterFit(
        center_pixel=(center_y, center_x),
        center_um=(center_y * dy_um, center_x * dx_um),
        sigma_pixel=(sigma_y, sigma_x),
        sigma_um=(sigma_y * dy_um, sigma_x * dx_um),
        background=background,
        peak_amplitude=peak_amplitude,
    )


def _elliptical_gaussian_amplitude(
    coordinates: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    background: float,
    peak_amplitude: float,
    center_y: float,
    center_x: float,
    sigma_y: float,
    sigma_x: float,
) -> npt.NDArray[np.float64]:
    y, x = coordinates
    exponent = -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2)
    return background + peak_amplitude * np.exp(exponent)


def _expanded_bounds(start: int, stop: int, size: int) -> tuple[int, int]:
    width = stop - start + 1
    padding = max(3, int(np.ceil(0.5 * width)))
    return max(0, start - padding), min(size, stop + padding + 1)


def _validate_open_unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must be a finite float in (0, 1), got {value}.")
    return result


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

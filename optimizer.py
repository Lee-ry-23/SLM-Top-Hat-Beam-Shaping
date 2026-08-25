from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
import torch

from array_helpers import center_crop_or_pad_array, expand_superpixel, normalize_power, validate_real_array
from initial_holograms import centered_pixel, lens_phase_hologram, random_hologram, zero_hologram
from loss_functions import build_support_mask, compute_benchmarks
from optical_planes import CameraPlane
from propagator import Propagator
from target_profiles import apply_psf_smoothing, build_line_target, build_rectangle_target
from validation_helpers import validate_array, validate_choice, validate_instance, validate_positive_float, validate_positive_int


@dataclass
class OptimizationResult:
    initial_hologram: npt.NDArray[np.float64]
    final_hologram: npt.NDArray[np.float64]
    full_resolution_hologram: npt.NDArray[np.float64]
    target_amplitude: npt.NDArray[np.float64]
    target_mask: npt.NDArray[np.float64]
    output_amplitude: npt.NDArray[np.float64]
    output_phase: npt.NDArray[np.float64]
    loss_history: list[float]
    iteration_loss_history: list[float]
    efficiency: float
    fidelity: float
    rms_error: float
    phase_error: float
    optimization_time_sec: float


class CGOptimizer:
    def __init__(
        self,
        propagator: Propagator,
        target_plane: CameraPlane | None = None,
        target_mask: npt.ArrayLike | None = None,
    ) -> None:
        self.propagator = validate_instance(propagator, Propagator, "propagator")
        self.target_plane: CameraPlane | None = None
        self.initial_hologram: npt.NDArray[np.float64] | None = None
        self.final_hologram: npt.NDArray[np.float64] | None = None
        self.loss_history: list[float] = []
        self.iteration_loss_history: list[float] = []
        self.optimization_result: scipy.optimize.OptimizeResult | None = None
        self.optimization_time_sec = 0.0
        self._target_amplitude: npt.NDArray[np.float64] | None = None
        self._target_phase: npt.NDArray[np.float64] | None = None
        self._target_mask: npt.NDArray[np.float64] | None = None
        self._last_loss: float | None = None
        if target_plane is not None:
            if target_mask is None:
                raise ValueError("target_mask must be provided when target_plane is provided to CGOptimizer.")
            self.set_target_plane(target_plane, target_mask)

    def set_target_plane(self, target_plane: CameraPlane, target_mask: npt.ArrayLike) -> None:
        self.target_plane = validate_instance(target_plane, CameraPlane, "target_plane")
        self._set_optimization_target_from_plane(self.target_plane, target_mask)

    def set_target_rectangle(
        self,
        width_x_um: float,
        width_y_um: float,
        psf_sigma_x_um: float | None,
        psf_sigma_y_um: float | None,
    ) -> None:
        target = build_rectangle_target(
            self.propagator.camera_plane.shape,
            self.propagator.camera_plane.x_axis_um,
            self.propagator.camera_plane.y_axis_um,
            width_x_um,
            width_y_um,
        )
        target = _smooth_target_if_requested(target, psf_sigma_x_um, psf_sigma_y_um, self.propagator.camera_plane.scale_um)
        self._set_camera_target(target, np.zeros_like(target), build_support_mask(target, 1e-3))

    def set_target_line(
        self,
        width_x_um: float,
        gaussian_diameter_y_um: float,
        psf_sigma_x_um: float | None,
        psf_sigma_y_um: float | None,
    ) -> None:
        target = build_line_target(
            self.propagator.camera_plane.shape,
            self.propagator.camera_plane.x_axis_um,
            self.propagator.camera_plane.y_axis_um,
            width_x_um,
            gaussian_diameter_y_um,
        )
        target = _smooth_target_if_requested(target, psf_sigma_x_um, psf_sigma_y_um, self.propagator.camera_plane.scale_um)
        self._set_camera_target(target, np.zeros_like(target), build_support_mask(target, 1e-3))

    def set_initial_hologram(self, method: str) -> None:
        selected_method = validate_choice(method, ("random", "zero", "lens"), "method")
        shape = self.propagator.slm_plane.shape
        if selected_method == "random":
            hologram = random_hologram(shape, None)
        elif selected_method == "zero":
            hologram = zero_hologram(shape)
        elif selected_method == "lens":
            hologram = lens_phase_hologram(shape, self.propagator.slm_plane.scale_um, 0.0, 0.0, 0.0, 0.0, centered_pixel(shape))
        else:
            raise ValueError(f"Unsupported initial hologram method {method}.")
        self.set_initial_hologram_array(hologram)

    def set_initial_hologram_array(self, hologram: npt.ArrayLike) -> None:
        hologram_array = validate_real_array(hologram, self.propagator.slm_plane.shape, "hologram")
        self.initial_hologram = np.mod(hologram_array, 2.0 * np.pi)
        self.final_hologram = None
        self.optimization_result = None
        self.propagator.slm_plane.set_hologram(self.initial_hologram)

    def optimize(
        self,
        maxiter: int,
        loss_scale: float,
        optimize_phase: bool,
    ) -> None:
        iterations = validate_positive_int(maxiter, "maxiter")
        scale = validate_positive_float(loss_scale, "loss_scale")
        if self.initial_hologram is None:
            raise RuntimeError("Initial hologram is not set. Call set_initial_hologram() or set_initial_hologram_array() first.")
        target_amplitude, target_phase, target_mask = self._require_optimization_target()

        self.loss_history = []
        self.iteration_loss_history = []
        self._last_loss = None
        self._loss_scale = scale
        self._optimize_phase = bool(optimize_phase)
        self._torch_cache = self._build_torch_cache(target_amplitude, target_phase, target_mask)

        def objective(hologram_1d: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
            loss_value, gradient = self.compute_loss_and_gradient(hologram_1d)
            self.loss_history.append(loss_value)
            self._last_loss = loss_value
            return loss_value, gradient

        def callback(_hologram_1d: npt.NDArray[np.float64]) -> None:
            if self._last_loss is None:
                raise RuntimeError("Optimizer callback was called before any loss evaluation.")
            self.iteration_loss_history.append(self._last_loss)

        start_time = perf_counter()
        result = scipy.optimize.minimize(
            objective,
            self.initial_hologram.reshape(-1),
            method="CG",
            jac=True,
            callback=callback,
            options={"maxiter": iterations, "disp": False},
        )
        self.optimization_time_sec = perf_counter() - start_time
        self.optimization_result = result
        self.final_hologram = np.mod(result.x.reshape(self.propagator.slm_plane.shape), 2.0 * np.pi)
        self.propagator.slm_plane.set_hologram(self.final_hologram)
        self.propagator.propagate("sft")
        del self._torch_cache

    def compute_loss_and_gradient(self, hologram_1d: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        if not hasattr(self, "_torch_cache"):
            target_amplitude, target_phase, target_mask = self._require_optimization_target()
            self._torch_cache = self._build_torch_cache(target_amplitude, target_phase, target_mask)
            self._loss_scale = 1.0
            self._optimize_phase = False

        cache = self._torch_cache
        dtype = cache["dtype"]
        device = cache["device"]
        slm_shape = self.propagator.slm_plane.shape
        padding_shape = self.propagator.padding_farfield.shape
        hologram_flat = torch.tensor(hologram_1d, dtype=dtype, device=device, requires_grad=True)
        hologram = hologram_flat.view(slm_shape)
        slm_field = cache["slm_amplitude"] * torch.exp(1j * (cache["slm_phase"] + hologram))
        padded_field = torch.zeros(padding_shape, dtype=torch.complex128, device=device)
        y0 = (padding_shape[0] - slm_shape[0]) // 2
        x0 = (padding_shape[1] - slm_shape[1]) // 2
        padded_field[y0:y0 + slm_shape[0], x0:x0 + slm_shape[1]] = cache["a0"] * slm_field
        output_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded_field)))
        output_amplitude = torch.abs(output_field)

        if self._optimize_phase:
            target_field = cache["target_amplitude"] * cache["target_mask"] * torch.exp(1j * cache["target_phase"])
            output_weighted_field = output_field * cache["target_mask"]
            numerator = torch.abs(torch.sum(torch.conj(target_field) * output_weighted_field)) ** 2
            denominator = torch.sum(torch.abs(target_field) ** 2) * torch.sum(torch.abs(output_weighted_field) ** 2)
            overlap = numerator / torch.clamp(denominator, min=torch.finfo(dtype).eps)
        else:
            numerator = torch.sum(cache["target_amplitude"] * output_amplitude * cache["target_mask"]) ** 2
            denominator = torch.sum((cache["target_amplitude"] * cache["target_mask"]) ** 2) * torch.sum((output_amplitude * cache["target_mask"]) ** 2)
            overlap = numerator / torch.clamp(denominator, min=torch.finfo(dtype).eps)
        loss = self._loss_scale * (1.0 - overlap) ** 2
        loss.backward()
        if hologram_flat.grad is None:
            raise RuntimeError("Torch did not produce a hologram gradient for the CG objective.")
        return float(loss.detach().cpu()), hologram_flat.grad.detach().cpu().numpy().reshape(-1)

    def get_final_hologram(self) -> npt.NDArray[np.float64]:
        if self.final_hologram is None:
            raise RuntimeError("Final hologram is not available. Run optimize() first.")
        return self.final_hologram.copy()

    def get_full_resolution_hologram(self) -> npt.NDArray[np.float64]:
        if self.final_hologram is None:
            raise RuntimeError("Final hologram is not available. Run optimize() first.")
        return np.mod(expand_superpixel(self.final_hologram, self.propagator.slm_plane.superpixel_size), 2.0 * np.pi)

    def get_result_summary(self) -> OptimizationResult:
        if self.initial_hologram is None or self.final_hologram is None:
            raise RuntimeError("Optimization result is not available. Run optimize() first.")
        target_amplitude, target_phase, target_mask = self._require_optimization_target()
        output_field = self.propagator.padding_farfield.get_field()
        metrics = compute_benchmarks(output_field, target_amplitude, target_phase, target_mask)
        return OptimizationResult(
            self.initial_hologram.copy(),
            self.final_hologram.copy(),
            self.get_full_resolution_hologram(),
            target_amplitude.copy(),
            target_mask.copy(),
            np.abs(output_field),
            np.angle(output_field),
            list(self.loss_history),
            list(self.iteration_loss_history),
            metrics["efficiency"],
            metrics["fidelity"],
            metrics["rms_error"],
            metrics["phase_error"],
            float(self.optimization_time_sec),
        )

    def plot_result_summary(self) -> plt.Figure:
        result = self.get_result_summary()
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
        _draw_image(axes[0, 0], result.initial_hologram, "Initial hologram", "twilight")
        _draw_image(axes[0, 1], result.final_hologram, "Final hologram", "twilight")
        _draw_image(axes[0, 2], result.target_amplitude**2 * result.target_mask, "Target intensity and mask", "inferno")
        _draw_image(axes[1, 0], result.output_amplitude**2, "Output intensity", "inferno")
        _draw_image(axes[1, 1], result.output_phase, "Output phase", "twilight")
        _draw_result_profiles(axes[1, 2], result)
        return fig

    def plot_loss_history(self) -> plt.Figure:
        if not self.loss_history:
            raise RuntimeError("Loss history is empty. Run optimize() first.")
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
        ax.plot(np.arange(len(self.loss_history)), self.loss_history, lw=1.8)
        ax.set_yscale("log")
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Loss")
        ax.set_title("CG loss history")
        ax.grid(True, alpha=0.25, linestyle="--")
        return fig

    def _set_camera_target(
        self,
        target_amplitude: npt.NDArray[np.float64],
        target_phase: npt.NDArray[np.float64],
        target_mask: npt.ArrayLike,
    ) -> None:
        self.target_plane = CameraPlane(
            self.propagator.camera_plane.shape,
            self.propagator.camera_plane.wavelength_nm,
            self.propagator.camera_plane.scale_um,
            self.propagator.camera_plane.camera_pixel_pitch_um,
            target_amplitude,
            target_phase,
        )
        self._set_optimization_target_from_plane(self.target_plane, target_mask)

    def _set_optimization_target_from_plane(self, target_plane: CameraPlane, target_mask: npt.ArrayLike) -> None:
        mask_camera = validate_array(target_mask, target_plane.shape, "target_mask")
        target_amplitude = center_crop_or_pad_array(target_plane.amplitude, self.propagator.padding_farfield.shape)
        target_phase = center_crop_or_pad_array(target_plane.phase, self.propagator.padding_farfield.shape)
        mask = center_crop_or_pad_array(mask_camera, self.propagator.padding_farfield.shape)
        if not np.any(mask > 0):
            raise ValueError("target_mask contains no active pixels after mapping to the optimization grid.")
        self._target_amplitude = normalize_power(target_amplitude * mask, self._input_power())
        self._target_phase = target_phase
        self._target_mask = mask

    def _require_optimization_target(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if self._target_amplitude is None or self._target_phase is None or self._target_mask is None:
            raise RuntimeError("Target is not set. Call set_target_plane(), set_target_rectangle(), or set_target_line() first.")
        return self._target_amplitude, self._target_phase, self._target_mask

    def _input_power(self) -> float:
        return float(np.sum(self.propagator.slm_plane.amplitude**2))

    def _build_torch_cache(
        self,
        target_amplitude: npt.NDArray[np.float64],
        target_phase: npt.NDArray[np.float64],
        target_mask: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        dtype = torch.float64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        padding_shape = self.propagator.padding_farfield.shape
        validate_array(target_amplitude, padding_shape, "target_amplitude")
        validate_array(target_phase, padding_shape, "target_phase")
        validate_array(target_mask, padding_shape, "target_mask")
        return {
            "dtype": dtype,
            "device": device,
            "a0": 1.0 / padding_shape[1],
            "slm_amplitude": torch.tensor(self.propagator.slm_plane.amplitude, dtype=dtype, device=device),
            "slm_phase": torch.tensor(self.propagator.slm_plane.phase, dtype=dtype, device=device),
            "target_amplitude": torch.tensor(target_amplitude, dtype=dtype, device=device),
            "target_phase": torch.tensor(target_phase, dtype=dtype, device=device),
            "target_mask": torch.tensor(target_mask, dtype=dtype, device=device),
        }


def _smooth_target_if_requested(
    target: npt.NDArray[np.float64],
    psf_sigma_x_um: float | None,
    psf_sigma_y_um: float | None,
    scale_um: tuple[float, float],
) -> npt.NDArray[np.float64]:
    sigma_x = 0.0 if psf_sigma_x_um is None else float(psf_sigma_x_um)
    sigma_y = 0.0 if psf_sigma_y_um is None else float(psf_sigma_y_um)
    if sigma_x == 0.0 and sigma_y == 0.0:
        return target
    return apply_psf_smoothing(target, sigma_x, sigma_y, scale_um)


def _draw_image(ax: plt.Axes, data: npt.NDArray[np.float64], title: str, cmap: str) -> None:
    image = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.82)


def _draw_result_profiles(ax: plt.Axes, result: OptimizationResult) -> None:
    center_y = result.output_amplitude.shape[0] // 2
    center_x = result.output_amplitude.shape[1] // 2
    x_active = result.target_mask[center_y, :] > 0
    y_active = result.target_mask[:, center_x] > 0
    if not np.any(x_active):
        raise ValueError("target_mask has no active pixels on the center x cut.")
    if not np.any(y_active):
        raise ValueError("target_mask has no active pixels on the center y cut.")

    x_axis = np.arange(result.output_amplitude.shape[1], dtype=float)[x_active]
    y_axis = np.arange(result.output_amplitude.shape[0], dtype=float)[y_active]
    target_x = result.target_amplitude[center_y, x_active] ** 2
    output_x = result.output_amplitude[center_y, x_active] ** 2
    target_y = result.target_amplitude[y_active, center_x] ** 2
    output_y = result.output_amplitude[y_active, center_x] ** 2

    ax.plot(x_axis, _normalize_profile(target_x), lw=2.0, label="target x")
    ax.plot(x_axis, _normalize_profile(output_x), lw=2.0, label="output x")
    ax.plot(y_axis, _normalize_profile(target_y), lw=2.0, ls="--", label="target y")
    ax.plot(y_axis, _normalize_profile(output_y), lw=2.0, ls="--", label="output y")
    ax.set_title("Mask-limited center cuts")
    ax.set_xlabel("pixel")
    ax.set_ylabel("normalized intensity")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8)


def _normalize_profile(profile: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    peak = float(np.max(profile))
    if peak <= 0:
        return profile.copy()
    return profile / peak

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
import torch

from .array_helpers import center_crop_or_pad_array, expand_superpixel, normalize_power, validate_real_array
from .loss_functions import compute_benchmarks
from .optical_planes import CameraPlane
from .propagator import Propagator
from .validation_helpers import validate_array, validate_choice, validate_instance, validate_positive_float, validate_positive_int


@dataclass
class DWChannelResult:
    target_amplitude: npt.NDArray[np.float64]
    target_mask: npt.NDArray[np.float64]
    output_amplitude: npt.NDArray[np.float64]
    output_phase: npt.NDArray[np.float64]
    overlap: float
    efficiency: float
    fidelity: float
    rms_error: float
    phase_error: float


@dataclass
class DWOptimizationResult:
    initial_hologram: npt.NDArray[np.float64]
    final_hologram: npt.NDArray[np.float64]
    full_resolution_hologram: npt.NDArray[np.float64]
    channel_1: DWChannelResult
    channel_2: DWChannelResult
    loss_history: list[float]
    iteration_loss_history: list[float]
    optimization_time_sec: float


class DWCGOptimizer:
    def __init__(
        self,
        propagator_1: Propagator,
        propagator_2: Propagator,
        target_plane_1: CameraPlane,
        target_mask_1: npt.ArrayLike,
        target_plane_2: CameraPlane,
        target_mask_2: npt.ArrayLike,
    ) -> None:
        self.propagator_1 = validate_instance(propagator_1, Propagator, "propagator_1")
        self.propagator_2 = validate_instance(propagator_2, Propagator, "propagator_2")
        _validate_shared_hologram_grid(self.propagator_1, self.propagator_2)
        self.target_plane_1 = validate_instance(target_plane_1, CameraPlane, "target_plane_1")
        self.target_plane_2 = validate_instance(target_plane_2, CameraPlane, "target_plane_2")
        self.initial_hologram: npt.NDArray[np.float64] | None = None
        self.final_hologram: npt.NDArray[np.float64] | None = None
        self.loss_history: list[float] = []
        self.iteration_loss_history: list[float] = []
        self.optimization_result: scipy.optimize.OptimizeResult | None = None
        self.optimization_time_sec = 0.0
        self._last_loss: float | None = None
        self._target_amplitude_1, self._target_phase_1, self._target_mask_1 = _build_channel_target(
            self.propagator_1,
            self.target_plane_1,
            target_mask_1,
            "channel_1",
        )
        self._target_amplitude_2, self._target_phase_2, self._target_mask_2 = _build_channel_target(
            self.propagator_2,
            self.target_plane_2,
            target_mask_2,
            "channel_2",
        )

    def set_initial_hologram_array(self, hologram: npt.ArrayLike) -> None:
        hologram_array = validate_real_array(hologram, self.propagator_1.slm_plane.shape, "hologram")
        self.initial_hologram = np.mod(hologram_array, 2.0 * np.pi)
        self.final_hologram = None
        self.optimization_result = None
        self._set_hologram_on_propagators(self.initial_hologram)

    def optimize(
        self,
        maxiter: int,
        loss_scale: float,
        optimize_phase: bool,
        method: str,
        channel_weight: float,
        exponential_rate: float,
    ) -> None:
        iterations = validate_positive_int(maxiter, "maxiter")
        scale = validate_positive_float(loss_scale, "loss_scale")
        loss_method = validate_choice(method, ("linear", "quadratic", "weighted_quadratic", "exponential"), "method")
        weight = _validate_unit_interval(channel_weight, "channel_weight")
        rate = validate_positive_float(exponential_rate, "exponential_rate")
        if self.initial_hologram is None:
            raise RuntimeError("Initial hologram is not set. Call set_initial_hologram_array() first.")

        self.loss_history = []
        self.iteration_loss_history = []
        self._last_loss = None
        self._loss_scale = scale
        self._optimize_phase = bool(optimize_phase)
        self._loss_method = loss_method
        self._channel_weight = weight
        self._exponential_rate = rate
        self._torch_cache_1 = self._build_torch_cache(
            self.propagator_1,
            self._target_amplitude_1,
            self._target_phase_1,
            self._target_mask_1,
        )
        self._torch_cache_2 = self._build_torch_cache(
            self.propagator_2,
            self._target_amplitude_2,
            self._target_phase_2,
            self._target_mask_2,
        )

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
        self.final_hologram = np.mod(result.x.reshape(self.propagator_1.slm_plane.shape), 2.0 * np.pi)
        self._set_hologram_on_propagators(self.final_hologram)
        self.propagator_1.propagate("sft")
        self.propagator_2.propagate("sft")
        del self._torch_cache_1
        del self._torch_cache_2

    def compute_loss_and_gradient(self, hologram_1d: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        if not hasattr(self, "_torch_cache_1") or not hasattr(self, "_torch_cache_2"):
            self._loss_scale = 1.0
            self._optimize_phase = False
            self._loss_method = "linear"
            self._channel_weight = 0.5
            self._exponential_rate = 1.0
            self._torch_cache_1 = self._build_torch_cache(
                self.propagator_1,
                self._target_amplitude_1,
                self._target_phase_1,
                self._target_mask_1,
            )
            self._torch_cache_2 = self._build_torch_cache(
                self.propagator_2,
                self._target_amplitude_2,
                self._target_phase_2,
                self._target_mask_2,
            )

        cache_1 = self._torch_cache_1
        dtype = cache_1["dtype"]
        device = cache_1["device"]
        hologram_flat = torch.tensor(hologram_1d, dtype=dtype, device=device, requires_grad=True)
        hologram = hologram_flat.view(self.propagator_1.slm_plane.shape)
        overlap_1 = _channel_overlap_torch(hologram, self._torch_cache_1, self._optimize_phase)
        overlap_2 = _channel_overlap_torch(hologram, self._torch_cache_2, self._optimize_phase)
        loss = _dual_wavelength_loss(
            overlap_1,
            overlap_2,
            self._loss_scale,
            self._loss_method,
            self._channel_weight,
            self._exponential_rate,
        )

        loss.backward()
        if hologram_flat.grad is None:
            raise RuntimeError("Torch did not produce a hologram gradient for the DWCG objective.")
        return float(loss.detach().cpu()), hologram_flat.grad.detach().cpu().numpy().reshape(-1)

    def get_final_hologram(self) -> npt.NDArray[np.float64]:
        if self.final_hologram is None:
            raise RuntimeError("Final hologram is not available. Run optimize() first.")
        return self.final_hologram.copy()

    def get_full_resolution_hologram(self) -> npt.NDArray[np.float64]:
        if self.final_hologram is None:
            raise RuntimeError("Final hologram is not available. Run optimize() first.")
        superpixel_size = self.propagator_1.slm_plane.superpixel_size
        return np.mod(expand_superpixel(self.final_hologram, superpixel_size), 2.0 * np.pi)

    def get_result_summary(self) -> DWOptimizationResult:
        if self.initial_hologram is None or self.final_hologram is None:
            raise RuntimeError("Optimization result is not available. Run optimize() first.")
        channel_1 = _build_channel_result(
            self.propagator_1,
            self._target_amplitude_1,
            self._target_phase_1,
            self._target_mask_1,
        )
        channel_2 = _build_channel_result(
            self.propagator_2,
            self._target_amplitude_2,
            self._target_phase_2,
            self._target_mask_2,
        )
        return DWOptimizationResult(
            self.initial_hologram.copy(),
            self.final_hologram.copy(),
            self.get_full_resolution_hologram(),
            channel_1,
            channel_2,
            list(self.loss_history),
            list(self.iteration_loss_history),
            float(self.optimization_time_sec),
        )

    def plot_result_summary(self) -> plt.Figure:
        result = self.get_result_summary()
        fig, axes = plt.subplots(3, 4, figsize=(18.0, 12.0), constrained_layout=True)
        _draw_image(axes[0, 0], result.initial_hologram, "Initial shared hologram", "twilight")
        _draw_image(axes[0, 1], result.final_hologram, "Final shared hologram", "twilight")
        _draw_loss_history(axes[0, 2], result.loss_history)
        axes[0, 3].axis("off")
        _draw_channel_summary_row(axes[1, :], result.channel_1, "Channel 1")
        _draw_channel_summary_row(axes[2, :], result.channel_2, "Channel 2")
        return fig

    def _set_hologram_on_propagators(self, hologram: npt.NDArray[np.float64]) -> None:
        self.propagator_1.slm_plane.set_hologram(hologram)
        self.propagator_2.slm_plane.set_hologram(hologram)

    def _build_torch_cache(
        self,
        propagator: Propagator,
        target_amplitude: npt.NDArray[np.float64],
        target_phase: npt.NDArray[np.float64],
        target_mask: npt.NDArray[np.float64],
    ) -> dict[str, object]:
        dtype = torch.float64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        padding_shape = propagator.padding_farfield.shape
        validate_array(target_amplitude, padding_shape, "target_amplitude")
        validate_array(target_phase, padding_shape, "target_phase")
        validate_array(target_mask, padding_shape, "target_mask")
        return {
            "dtype": dtype,
            "device": device,
            "a0": 1.0 / padding_shape[1],
            "slm_shape": propagator.slm_plane.shape,
            "padding_shape": padding_shape,
            "slm_amplitude": torch.tensor(propagator.slm_plane.amplitude, dtype=dtype, device=device),
            "slm_phase": torch.tensor(propagator.slm_plane.phase, dtype=dtype, device=device),
            "target_amplitude": torch.tensor(target_amplitude, dtype=dtype, device=device),
            "target_phase": torch.tensor(target_phase, dtype=dtype, device=device),
            "target_mask": torch.tensor(target_mask, dtype=dtype, device=device),
        }



def _dual_wavelength_loss(
    overlap_1: torch.Tensor,
    overlap_2: torch.Tensor,
    loss_scale: float,
    method: str,
    channel_weight: float,
    exponential_rate: float,
) -> torch.Tensor:
    if method == "linear":
        unscaled_loss = (2.0 - overlap_1 - overlap_2) ** 2
    elif method == "quadratic":
        unscaled_loss = ((1.0 - overlap_1) ** 2 + (1.0 - overlap_2) ** 2) / 2.0
    elif method == "weighted_quadratic":
        unscaled_loss = channel_weight * (1.0 - overlap_1) ** 2 + (1.0 - channel_weight) * (1.0 - overlap_2) ** 2
    elif method == "exponential":
        base = torch.as_tensor(2.0, dtype=overlap_1.dtype, device=overlap_1.device)
        unscaled_loss = torch.pow(base, exponential_rate * (1.0 - overlap_1)) + torch.pow(base, exponential_rate * (1.0 - overlap_2))
    else:
        raise ValueError(f"Unsupported dual-wavelength loss method {method}.")
    return float(loss_scale) * unscaled_loss


def _validate_unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1], got {value}.")
    return result

def _validate_shared_hologram_grid(propagator_1: Propagator, propagator_2: Propagator) -> None:
    if propagator_1.slm_plane.shape != propagator_2.slm_plane.shape:
        raise ValueError(
            "Both propagators must use the same SLM shape for a shared hologram, "
            f"got {propagator_1.slm_plane.shape} and {propagator_2.slm_plane.shape}."
        )
    if propagator_1.slm_plane.superpixel_size != propagator_2.slm_plane.superpixel_size:
        raise ValueError(
            "Both propagators must use the same superpixel_size for a shared hologram, "
            f"got {propagator_1.slm_plane.superpixel_size} and {propagator_2.slm_plane.superpixel_size}."
        )


def _build_channel_target(
    propagator: Propagator,
    target_plane: CameraPlane,
    target_mask: npt.ArrayLike,
    channel_name: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    mask_camera = validate_array(target_mask, target_plane.shape, f"{channel_name}.target_mask")
    target_amplitude = center_crop_or_pad_array(target_plane.amplitude, propagator.padding_farfield.shape)
    target_phase = center_crop_or_pad_array(target_plane.phase, propagator.padding_farfield.shape)
    mask = center_crop_or_pad_array(mask_camera, propagator.padding_farfield.shape)
    if not np.any(mask > 0):
        raise ValueError(f"{channel_name}.target_mask contains no active pixels after mapping to the optimization grid.")
    target_power = float(np.sum(propagator.slm_plane.amplitude**2))
    return normalize_power(target_amplitude * mask, target_power), target_phase, mask


def _channel_overlap_torch(
    hologram: torch.Tensor,
    cache: dict[str, object],
    optimize_phase: bool,
) -> torch.Tensor:
    slm_shape = cache["slm_shape"]
    padding_shape = cache["padding_shape"]
    slm_amplitude = cache["slm_amplitude"]
    slm_phase = cache["slm_phase"]
    target_amplitude = cache["target_amplitude"]
    target_phase = cache["target_phase"]
    target_mask = cache["target_mask"]
    dtype = cache["dtype"]
    device = cache["device"]
    slm_field = slm_amplitude * torch.exp(1j * (slm_phase + hologram))
    padded_field = torch.zeros(padding_shape, dtype=torch.complex128, device=device)
    y0 = (padding_shape[0] - slm_shape[0]) // 2
    x0 = (padding_shape[1] - slm_shape[1]) // 2
    padded_field[y0:y0 + slm_shape[0], x0:x0 + slm_shape[1]] = cache["a0"] * slm_field
    output_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded_field)))
    output_amplitude = torch.abs(output_field)
    if optimize_phase:
        target_field = target_amplitude * target_mask * torch.exp(1j * target_phase)
        output_weighted_field = output_field * target_mask
        numerator = torch.abs(torch.sum(torch.conj(target_field) * output_weighted_field)) ** 2
        denominator = torch.sum(torch.abs(target_field) ** 2) * torch.sum(torch.abs(output_weighted_field) ** 2)
    else:
        numerator = torch.sum(target_amplitude * output_amplitude * target_mask) ** 2
        denominator = torch.sum((target_amplitude * target_mask) ** 2) * torch.sum((output_amplitude * target_mask) ** 2)
    return numerator / torch.clamp(denominator, min=torch.finfo(dtype).eps)


def _build_channel_result(
    propagator: Propagator,
    target_amplitude: npt.NDArray[np.float64],
    target_phase: npt.NDArray[np.float64],
    target_mask: npt.NDArray[np.float64],
) -> DWChannelResult:
    output_field = propagator.padding_farfield.get_field()
    metrics = compute_benchmarks(output_field, target_amplitude, target_phase, target_mask)
    overlap = _overlap_numpy(output_field, target_amplitude, target_phase, target_mask)
    return DWChannelResult(
        target_amplitude.copy(),
        target_mask.copy(),
        np.abs(output_field),
        np.angle(output_field),
        overlap,
        metrics["efficiency"],
        metrics["fidelity"],
        metrics["rms_error"],
        metrics["phase_error"],
    )


def _overlap_numpy(
    output_field: npt.NDArray[np.complex128],
    target_amplitude: npt.NDArray[np.float64],
    target_phase: npt.NDArray[np.float64],
    target_mask: npt.NDArray[np.float64],
) -> float:
    target_field = target_amplitude * target_mask * np.exp(1j * target_phase)
    output_weighted_field = output_field * target_mask
    numerator = float(np.abs(np.sum(np.conj(target_field) * output_weighted_field)) ** 2)
    denominator = float(np.sum(np.abs(target_field) ** 2) * np.sum(np.abs(output_weighted_field) ** 2))
    if denominator <= 0:
        raise ValueError("Cannot compute channel overlap because the denominator is zero.")
    return numerator / denominator


def _draw_channel_summary_row(axes: npt.NDArray[object], channel: DWChannelResult, label: str) -> None:
    _draw_image(axes[0], channel.target_amplitude**2 * channel.target_mask, f"{label} target and mask", "inferno")
    _draw_image(axes[1], channel.output_amplitude**2, f"{label} output intensity", "inferno")
    _draw_image(axes[2], channel.output_phase, f"{label} output phase", "twilight")
    _draw_channel_profiles(axes[3], channel, f"{label} mask-limited cuts")


def _draw_image(ax: plt.Axes, data: npt.NDArray[np.float64], title: str, cmap: str) -> None:
    image = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.82)


def _draw_loss_history(ax: plt.Axes, loss_history: list[float]) -> None:
    if not loss_history:
        ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.plot(np.arange(len(loss_history)), loss_history, lw=1.8)
    ax.set_yscale("log")
    ax.set_title("DWCG loss history")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25, linestyle="--")


def _draw_channel_profiles(ax: plt.Axes, channel: DWChannelResult, title: str) -> None:
    center_y = channel.output_amplitude.shape[0] // 2
    center_x = channel.output_amplitude.shape[1] // 2
    x_active = channel.target_mask[center_y, :] > 0
    y_active = channel.target_mask[:, center_x] > 0
    if not np.any(x_active):
        raise ValueError(f"{title}: target_mask has no active pixels on the center x cut.")
    if not np.any(y_active):
        raise ValueError(f"{title}: target_mask has no active pixels on the center y cut.")
    x_axis = np.arange(channel.output_amplitude.shape[1], dtype=float)[x_active]
    y_axis = np.arange(channel.output_amplitude.shape[0], dtype=float)[y_active]
    target_x = channel.target_amplitude[center_y, x_active] ** 2
    output_x = channel.output_amplitude[center_y, x_active] ** 2
    target_y = channel.target_amplitude[y_active, center_x] ** 2
    output_y = channel.output_amplitude[y_active, center_x] ** 2
    ax.plot(x_axis, _normalize_profile(target_x), lw=2.0, label="target x")
    ax.plot(x_axis, _normalize_profile(output_x), lw=2.0, label="output x")
    ax.plot(y_axis, _normalize_profile(target_y), lw=2.0, ls="--", label="target y")
    ax.plot(y_axis, _normalize_profile(output_y), lw=2.0, ls="--", label="output y")
    ax.set_title(title)
    ax.set_xlabel("pixel")
    ax.set_ylabel("normalized intensity")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8)


def _normalize_profile(profile: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    peak = float(np.max(profile))
    if peak <= 0:
        return profile.copy()
    return profile / peak

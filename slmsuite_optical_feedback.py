from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import FeedbackHologram
from slmsuite.holography.toolbox.phase import blaze

from functions import build_target, build_weighting_mask, get_focal_plane_axes_um, get_plot_radius


class FeedbackFitResult(TypedDict):
    center_xy_px: tuple[float, float]
    angle_deg: float
    cost: float
    target_amplitude: npt.NDArray[np.float64]
    weighting_mask: npt.NDArray[np.float64]
    extracted_image: npt.NDArray[np.float64]


def capture_slmsuite_image(fs: FourierSLM) -> npt.NDArray[np.float64]:
    image = np.asarray(fs.cam.get_image(), dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Camera image must be 2D, got shape {image.shape}.")
    return image


def preview_feedback_zoom(
    cfg,
    camera_image: npt.ArrayLike,
    center_xy_px: tuple[float, float],
    crop_shape_yx_px: tuple[int, int],
) -> None:
    image = _rotate_camera_image_for_feedback(cfg, camera_image)
    crop = _crop_image(image, center_xy_px, crop_shape_yx_px)
    crop_y0, crop_x0 = _crop_origin(image.shape, center_xy_px, crop_shape_yx_px)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].imshow(image, origin="lower", cmap="magma")
    axes[0].scatter([center_xy_px[0]], [center_xy_px[1]], s=36, c="cyan", marker="+")
    axes[0].add_patch(
        Rectangle(
            (crop_x0, crop_y0),
            crop_shape_yx_px[1],
            crop_shape_yx_px[0],
            fill=False,
            edgecolor="cyan",
            linewidth=1.6,
        )
    )
    axes[0].set_title("Camera image and selected zoom")
    axes[0].set_xlabel("camera x (px)")
    axes[0].set_ylabel("camera y (px)")

    axes[1].imshow(crop, origin="lower", cmap="magma")
    axes[1].scatter(
        [center_xy_px[0] - crop_x0],
        [center_xy_px[1] - crop_y0],
        s=42,
        c="cyan",
        marker="+",
    )
    axes[1].set_title("Zoom used for feedback calibration")
    axes[1].set_xlabel("zoom x (px)")
    axes[1].set_ylabel("zoom y (px)")
    plt.show()


def fit_feedback_extraction(
    cfg,
    camera_image: npt.ArrayLike,
    center_xy_px: tuple[float, float],
    crop_shape_yx_px: tuple[int, int],
    target_amplitude: npt.ArrayLike | None,
) -> FeedbackFitResult:
    image = _prepare_camera_image(cfg, camera_image)
    _crop_image(image, center_xy_px, crop_shape_yx_px)

    target, mask = _target_and_mask(cfg, target_amplitude)
    target_active = target * mask
    target_norm = _normalized_active_values(target_active, mask)

    initial_angle = float(cfg.feedback_angle_guess_deg)
    angle_radius = float(cfg.feedback_angle_search_radius_deg)
    center_bounds = _center_search_bounds(image.shape, center_xy_px, crop_shape_yx_px, float(cfg.feedback_center_search_radius_px))

    def objective(params: npt.NDArray[np.float64]) -> float:
        return _feedback_pose_cost(cfg, image, mask, target_norm, (float(params[0]), float(params[1])), float(params[2]))

    coarse_center = _fit_center_in_zoom(
        cfg,
        image,
        mask,
        target_norm,
        center_bounds,
        initial_angle,
    )

    result = scipy.optimize.minimize(
        objective,
        x0=np.array([coarse_center[0], coarse_center[1], initial_angle], dtype=float),
        method="L-BFGS-B",
        bounds=[
            center_bounds[0],
            center_bounds[1],
            (initial_angle - angle_radius, initial_angle + angle_radius),
        ],
    )
    if not result.success or float(result.fun) >= 1e11:
        raise RuntimeError(
            "Optical feedback pattern fit failed: "
            f"message={result.message}, cost={float(result.fun):.3e}, "
            f"initial_center_xy_px={center_xy_px}, crop_shape_yx_px={crop_shape_yx_px}."
        )

    fitted_center = (float(result.x[0]), float(result.x[1]))
    fitted_angle = float(result.x[2])
    extracted_image = extract_feedback_image(cfg, camera_image, fitted_center, fitted_angle, target)

    return {
        "center_xy_px": fitted_center,
        "angle_deg": fitted_angle,
        "cost": float(result.fun),
        "target_amplitude": target,
        "weighting_mask": mask,
        "extracted_image": extracted_image,
    }


def plot_feedback_extraction(
    cfg,
    camera_image: npt.ArrayLike,
    fit_result: FeedbackFitResult,
    crop_shape_yx_px: tuple[int, int],
) -> None:
    image = _rotate_camera_image_for_feedback(cfg, camera_image)
    center_xy_px = fit_result["center_xy_px"]
    crop = _crop_image(image, center_xy_px, crop_shape_yx_px)
    crop_y0, crop_x0 = _crop_origin(image.shape, center_xy_px, crop_shape_yx_px)
    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)
    focal_extent = _axis_extent_from_coords(focal_x_um, focal_y_um)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    axes[0, 0].imshow(image, origin="lower", cmap="magma")
    axes[0, 0].scatter([center_xy_px[0]], [center_xy_px[1]], s=42, c="cyan", marker="+")
    axes[0, 0].add_patch(
        Rectangle(
            (crop_x0, crop_y0),
            crop_shape_yx_px[1],
            crop_shape_yx_px[0],
            fill=False,
            edgecolor="cyan",
            linewidth=1.5,
        )
    )
    axes[0, 0].set_title("Fitted camera position")
    axes[0, 0].set_xlabel("camera x (px)")
    axes[0, 0].set_ylabel("camera y (px)")

    axes[0, 1].imshow(crop, origin="lower", cmap="magma")
    axes[0, 1].scatter([center_xy_px[0] - crop_x0], [center_xy_px[1] - crop_y0], s=42, c="cyan", marker="+")
    axes[0, 1].set_title(f"Zoom, fitted angle = {fit_result['angle_deg']:.2f} deg")
    axes[0, 1].set_xlabel("zoom x (px)")
    axes[0, 1].set_ylabel("zoom y (px)")

    im_target = axes[1, 0].imshow(
        fit_result["target_amplitude"],
        origin="lower",
        cmap="viridis",
        extent=focal_extent,
        aspect="auto",
    )
    axes[1, 0].contour(focal_x_um, focal_y_um, fit_result["weighting_mask"] > 0, levels=[0.5], colors="white", linewidths=0.8)
    axes[1, 0].set_title("Target and feedback mask")
    axes[1, 0].set_xlabel("Focal-plane x (um)")
    axes[1, 0].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axes[1, 0], focal_x_um, focal_y_um)
    fig.colorbar(im_target, ax=axes[1, 0], shrink=0.82)

    im_feedback = axes[1, 1].imshow(
        fit_result["extracted_image"],
        origin="lower",
        cmap="viridis",
        extent=focal_extent,
        aspect="auto",
    )
    axes[1, 1].contour(focal_x_um, focal_y_um, fit_result["weighting_mask"] > 0, levels=[0.5], colors="white", linewidths=0.8)
    axes[1, 1].set_title("Extracted experimental feedback")
    axes[1, 1].set_xlabel("Focal-plane x (um)")
    axes[1, 1].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axes[1, 1], focal_x_um, focal_y_um)
    fig.colorbar(im_feedback, ax=axes[1, 1], shrink=0.82)

    fig.suptitle(f"Optical feedback extraction, fit cost = {fit_result['cost']:.3e}")
    plt.show()


def extract_feedback_image(
    cfg,
    camera_image: npt.ArrayLike,
    center_xy_px: tuple[float, float],
    angle_deg: float,
    target_amplitude: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    image = _prepare_camera_image(cfg, camera_image)
    target, mask = _target_and_mask(cfg, target_amplitude)
    coords_y, coords_x = _feedback_sampling_coordinates(cfg, center_xy_px, angle_deg)

    if not _active_region_is_inside_camera(image.shape, coords_y, coords_x, mask):
        raise ValueError(
            "Feedback mask maps outside the camera image. "
            f"center_xy_px={center_xy_px}, angle_deg={angle_deg}, "
            f"camera_shape_yx={image.shape}."
        )

    sampled = _sample_camera_image(image, coords_y, coords_x)
    return _camera_signal_to_feedback_amplitude(cfg, sampled) * mask


def make_slmsuite_feedback_image_func(
    cfg,
    fs: FourierSLM,
    fit_result: FeedbackFitResult,
) -> Callable[[], npt.NDArray[np.float64]]:
    center_xy_px = fit_result["center_xy_px"]
    angle_deg = fit_result["angle_deg"]
    target_amplitude = fit_result["target_amplitude"]

    def get_image_func() -> npt.NDArray[np.float64]:
        camera_image = capture_slmsuite_image(fs)
        return extract_feedback_image(cfg, camera_image, center_xy_px, angle_deg, target_amplitude)

    return get_image_func


def apply_feedback_hologram_with_blaze(
    fs: FourierSLM,
    slm_pattern: npt.ArrayLike,
    blaze_vector: tuple[float, float],
) -> FeedbackHologram:
    hologram = FeedbackHologram(shape=fs.slm.shape, cameraslm=fs)
    blaze_phase = blaze(grid=fs.slm, vector=blaze_vector)
    hologram.reset_phase(np.mod(np.asarray(slm_pattern, dtype=float) + blaze_phase, 2 * np.pi))
    return hologram


def derive_get_image_func_from_feedback_hologram(
    cfg,
    fs: FourierSLM,
    slm_pattern: npt.ArrayLike,
    blaze_vector: tuple[float, float],
) -> Callable[[], npt.NDArray[np.float64]]:
    if cfg.feedback_center_xy_px is None:
        raise ValueError("cfg.feedback_center_xy_px must be set before deriving optical feedback.")

    apply_feedback_hologram_with_blaze(fs, slm_pattern, blaze_vector)
    camera_image = capture_slmsuite_image(fs)
    fit_result = fit_feedback_extraction(
        cfg,
        camera_image,
        cfg.feedback_center_xy_px,
        cfg.feedback_crop_shape_yx_px,
        None,
    )
    return make_slmsuite_feedback_image_func(cfg, fs, fit_result)


def plot_feedback_profiles(
    cfg,
    get_image_func: Callable[[], npt.NDArray[np.float64]],
    target_amplitude: npt.ArrayLike | None,
    save_path: str | None,
    show: bool,
) -> plt.Figure:
    feedback_amplitude = np.asarray(get_image_func(), dtype=float)
    target, mask = _target_and_mask(cfg, target_amplitude)

    if feedback_amplitude.shape != target.shape:
        raise ValueError(
            "Feedback image shape does not match the optimization grid. "
            f"feedback_shape={feedback_amplitude.shape}, target_shape={target.shape}. "
            "Do not resize feedback images here because it changes the physical x/y ratio."
        )

    feedback_intensity = _normalize_image((np.clip(feedback_amplitude, 0.0, None) * mask) ** 2)
    target_intensity = _normalize_image((np.clip(target, 0.0, None) * mask) ** 2)

    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)
    radius_um = get_plot_radius(cfg)
    feedback_crop, bounds = _crop_roi(feedback_intensity, focal_x_um, focal_y_um, radius_um)
    target_crop, _ = _crop_roi(target_intensity, focal_x_um, focal_y_um, radius_um)

    x_min, x_max, y_min, y_max = bounds
    crop_x_um = focal_x_um[x_min:x_max]
    crop_y_um = focal_y_um[y_min:y_max]
    extent = _axis_extent_from_coords(crop_x_um, crop_y_um)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle("Experimental Optical Feedback Profiles", fontsize=15, fontweight="bold")

    im0 = axes[0].imshow(target_crop, origin="lower", cmap="inferno", vmin=0, vmax=1, extent=extent, aspect="auto")
    axes[0].set_title("Target ROI Intensity")
    axes[0].set_xlabel("Focal-plane x (um)")
    axes[0].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axes[0], crop_x_um, crop_y_um)
    fig.colorbar(im0, ax=axes[0], shrink=0.84)

    im1 = axes[1].imshow(feedback_crop, origin="lower", cmap="inferno", vmin=0, vmax=1, extent=extent, aspect="auto")
    axes[1].set_title("Experimental ROI Intensity")
    axes[1].set_xlabel("Focal-plane x (um)")
    axes[1].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axes[1], crop_x_um, crop_y_um)
    fig.colorbar(im1, ax=axes[1], shrink=0.84)

    _plot_center_profiles(
        axes[2],
        feedback_crop,
        target_crop,
        crop_x_um,
        crop_y_um,
        "Target vs Experimental Profiles",
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _crop_roi(
    data: npt.NDArray[np.float64],
    x_axis: npt.NDArray[np.float64],
    y_axis: npt.NDArray[np.float64],
    radius_um: float,
) -> tuple[npt.NDArray[np.float64], tuple[int, int, int, int]]:
    x_mask = np.abs(x_axis) <= radius_um
    y_mask = np.abs(y_axis) <= radius_um
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    if x_indices.size == 0 or y_indices.size == 0:
        raise ValueError(
            "Feedback profile crop contains no pixels. "
            f"radius_um={radius_um}, x_range=({x_axis[0]}, {x_axis[-1]}), "
            f"y_range=({y_axis[0]}, {y_axis[-1]})."
        )

    x_min, x_max = int(x_indices[0]), int(x_indices[-1] + 1)
    y_min, y_max = int(y_indices[0]), int(y_indices[-1] + 1)
    return data[y_min:y_max, x_min:x_max], (x_min, x_max, y_min, y_max)


def _normalize_image(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    peak = float(np.max(data))
    if peak <= 0:
        return data
    return data / peak


def _axis_extent_from_coords(
    x_axis: npt.NDArray[np.float64],
    y_axis: npt.NDArray[np.float64],
) -> list[float]:
    return [float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])]


def _apply_square_axes(
    ax: plt.Axes,
    x_axis: npt.NDArray[np.float64],
    y_axis: npt.NDArray[np.float64],
) -> None:
    half_span = max(float(np.max(np.abs(x_axis))), float(np.max(np.abs(y_axis))))
    ax.set_xlim(-half_span, half_span)
    ax.set_ylim(-half_span, half_span)
    ax.set_aspect("equal", adjustable="box")


def _plot_center_profiles(
    ax: plt.Axes,
    experimental_intensity: npt.NDArray[np.float64],
    target_intensity: npt.NDArray[np.float64],
    x_axis_um: npt.NDArray[np.float64],
    y_axis_um: npt.NDArray[np.float64],
    title: str,
) -> None:
    center_y = experimental_intensity.shape[0] // 2
    center_x = experimental_intensity.shape[1] // 2
    ax.plot(x_axis_um, target_intensity[center_y, :], lw=2.2, color="#1f77b4", label="Target x-cut")
    ax.plot(x_axis_um, experimental_intensity[center_y, :], lw=2.2, color="#d62728", label="Experimental x-cut")
    ax.plot(y_axis_um, target_intensity[:, center_x], lw=1.9, ls="--", color="#2ca02c", label="Target y-cut")
    ax.plot(y_axis_um, experimental_intensity[:, center_x], lw=1.9, ls="--", color="#9467bd", label="Experimental y-cut")
    ax.set_title(title)
    ax.set_xlabel("Position (um)")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(frameon=True, fontsize=8, loc="best")


def _as_float_image(camera_image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    image = np.asarray(camera_image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Camera image must be 2D, got shape {image.shape}.")
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _rotate_camera_image_for_feedback(cfg, camera_image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    image = _as_float_image(camera_image)
    rotation_deg = int(cfg.feedback_camera_rotation_deg)
    if rotation_deg % 90 != 0:
        raise ValueError(
            "feedback_camera_rotation_deg must be a multiple of 90 degrees. "
            f"Got {cfg.feedback_camera_rotation_deg}."
        )

    return np.rot90(image, k=(rotation_deg // 90) % 4)


def _prepare_camera_image(cfg, camera_image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    image = _rotate_camera_image_for_feedback(cfg, camera_image)
    background_percentile = float(cfg.feedback_background_percentile)
    background = float(np.percentile(image, background_percentile))
    image = np.clip(image - background, 0.0, None)
    if np.max(image) <= 0:
        raise ValueError(
            "Camera image contains no positive signal after background subtraction. "
            f"background_percentile={background_percentile}."
        )
    return image


def _crop_origin(
    image_shape_yx: tuple[int, int],
    center_xy_px: tuple[float, float],
    crop_shape_yx_px: tuple[int, int],
) -> tuple[int, int]:
    crop_h, crop_w = crop_shape_yx_px
    center_x, center_y = center_xy_px
    y0 = int(round(center_y - crop_h / 2))
    x0 = int(round(center_x - crop_w / 2))
    if y0 < 0 or x0 < 0 or y0 + crop_h > image_shape_yx[0] or x0 + crop_w > image_shape_yx[1]:
        raise ValueError(
            "Feedback zoom is outside the camera image. "
            f"center_xy_px={center_xy_px}, crop_shape_yx_px={crop_shape_yx_px}, "
            f"camera_shape_yx={image_shape_yx}."
        )
    return y0, x0


def _crop_image(
    image: npt.NDArray[np.float64],
    center_xy_px: tuple[float, float],
    crop_shape_yx_px: tuple[int, int],
) -> npt.NDArray[np.float64]:
    y0, x0 = _crop_origin(image.shape, center_xy_px, crop_shape_yx_px)
    crop_h, crop_w = crop_shape_yx_px
    return image[y0:y0 + crop_h, x0:x0 + crop_w]


def _center_search_bounds(
    image_shape_yx: tuple[int, int],
    center_xy_px: tuple[float, float],
    crop_shape_yx_px: tuple[int, int],
    center_radius_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    crop_y0, crop_x0 = _crop_origin(image_shape_yx, center_xy_px, crop_shape_yx_px)
    crop_h, crop_w = crop_shape_yx_px
    center_x, center_y = center_xy_px
    x_bounds = (
        max(float(crop_x0), center_x - center_radius_px),
        min(float(crop_x0 + crop_w - 1), center_x + center_radius_px),
    )
    y_bounds = (
        max(float(crop_y0), center_y - center_radius_px),
        min(float(crop_y0 + crop_h - 1), center_y + center_radius_px),
    )
    if x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]:
        raise ValueError(
            "Invalid feedback center search bounds. "
            f"center_xy_px={center_xy_px}, crop_shape_yx_px={crop_shape_yx_px}, "
            f"center_radius_px={center_radius_px}."
        )
    return x_bounds, y_bounds


def _fit_center_in_zoom(
    cfg,
    image: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    target_norm: npt.NDArray[np.float64],
    center_bounds: tuple[tuple[float, float], tuple[float, float]],
    angle_deg: float,
) -> tuple[float, float]:
    x_bounds, y_bounds = center_bounds
    step_px = float(cfg.feedback_center_coarse_step_px)
    if step_px <= 0:
        raise ValueError(f"feedback_center_coarse_step_px must be positive, got {step_px}.")

    x_candidates = _candidate_axis(x_bounds, step_px)
    y_candidates = _candidate_axis(y_bounds, step_px)
    best_center = (float(x_candidates[0]), float(y_candidates[0]))
    best_cost = np.inf

    for candidate_y in y_candidates:
        for candidate_x in x_candidates:
            candidate_center = (float(candidate_x), float(candidate_y))
            cost = _feedback_pose_cost(cfg, image, mask, target_norm, candidate_center, angle_deg)
            if cost < best_cost:
                best_cost = cost
                best_center = candidate_center

    if not np.isfinite(best_cost) or best_cost >= 1e11:
        raise RuntimeError(
            "Coarse feedback center fit failed inside the selected zoom. "
            f"center_bounds={center_bounds}, angle_deg={angle_deg}, best_cost={best_cost:.3e}."
        )
    return best_center


def _candidate_axis(bounds: tuple[float, float], step_px: float) -> npt.NDArray[np.float64]:
    lower, upper = bounds
    values = np.arange(lower, upper + 0.5 * step_px, step_px, dtype=float)
    if values.size == 0 or values[-1] < upper:
        values = np.append(values, upper)
    return np.unique(np.clip(values, lower, upper))


def _feedback_pose_cost(
    cfg,
    image: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    target_norm: npt.NDArray[np.float64],
    center_xy_px: tuple[float, float],
    angle_deg: float,
) -> float:
    coords_y, coords_x = _feedback_sampling_coordinates(cfg, center_xy_px, angle_deg)
    if not _active_region_is_inside_camera(image.shape, coords_y, coords_x, mask):
        return 1e12

    sampled = _sample_camera_image(image, coords_y, coords_x)
    sampled = _camera_signal_to_feedback_amplitude(cfg, sampled) * mask
    try:
        sampled_norm = _normalized_active_values(sampled, mask)
    except ValueError:
        return 1e12
    return float(np.mean((sampled_norm - target_norm) ** 2))


def _target_and_mask(
    cfg,
    target_amplitude: npt.ArrayLike | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if target_amplitude is None:
        target = np.asarray(build_target(cfg), dtype=float)
    else:
        target = np.asarray(target_amplitude, dtype=float)
        expected_shape = (cfg.NTy, cfg.NTx)
        if target.shape != expected_shape:
            raise ValueError(
                "Feedback target amplitude must already be on the optimization grid. "
                f"target_shape={target.shape}, expected_shape={expected_shape}. "
                "Use the same physical focal-plane grid as cfg instead of resizing here."
            )

    if np.max(np.abs(target)) <= 0:
        raise ValueError("Feedback target amplitude contains no positive signal.")

    mask = np.asarray(build_weighting_mask(cfg, target), dtype=float)
    return target, mask


def _feedback_sampling_coordinates(
    cfg,
    center_xy_px: tuple[float, float],
    angle_deg: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)
    x_um, y_um = np.meshgrid(focal_x_um, focal_y_um, indexing="xy")
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    center_x_px, center_y_px = center_xy_px

    camera_x_px = center_x_px + (cos_t * x_um - sin_t * y_um) / cfg.camera_pixel_size_um
    camera_y_px = center_y_px + (sin_t * x_um + cos_t * y_um) / cfg.camera_pixel_size_um
    return camera_y_px, camera_x_px


def _active_region_is_inside_camera(
    image_shape_yx: tuple[int, int],
    coords_y: npt.NDArray[np.float64],
    coords_x: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
) -> bool:
    active = mask > 0
    if not np.any(active):
        raise ValueError("Feedback weighting mask has no active pixels.")

    active_x = coords_x[active]
    active_y = coords_y[active]
    return bool(
        np.min(active_x) >= 0
        and np.max(active_x) <= image_shape_yx[1] - 1
        and np.min(active_y) >= 0
        and np.max(active_y) <= image_shape_yx[0] - 1
    )


def _sample_camera_image(
    image: npt.NDArray[np.float64],
    coords_y: npt.NDArray[np.float64],
    coords_x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    sampled = map_coordinates(image, [coords_y, coords_x], order=1, mode="constant", cval=0.0)
    return np.asarray(sampled, dtype=float)


def _camera_signal_to_feedback_amplitude(
    cfg,
    camera_signal: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    signal = np.clip(camera_signal, 0.0, None)
    if cfg.feedback_camera_image_is_intensity:
        return np.sqrt(signal)
    return signal


def _normalized_active_values(
    image: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    values = np.asarray(image[mask > 0], dtype=float)
    norm = float(np.sqrt(np.sum(values**2)))
    if norm <= 0:
        raise ValueError("Active feedback region contains no positive signal.")
    return values / norm

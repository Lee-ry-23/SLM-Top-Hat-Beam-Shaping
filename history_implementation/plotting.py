import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#B0B0B0",
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def _crop_roi(data, x_axis, y_axis, radius_um):
    x_mask = np.abs(x_axis) <= radius_um
    y_mask = np.abs(y_axis) <= radius_um
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    x_min, x_max = x_indices[0], x_indices[-1] + 1
    y_min, y_max = y_indices[0], y_indices[-1] + 1
    return data[y_min:y_max, x_min:x_max], (x_min, x_max, y_min, y_max)


def _normalize_image(data):
    peak = np.max(data)
    return data / peak if peak > 0 else data


def _axis_extent_from_coords(x_axis, y_axis):
    return [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]


def _square_limits(x_axis, y_axis):
    half_span = max(np.max(np.abs(x_axis)), np.max(np.abs(y_axis)))
    return (-half_span, half_span), (-half_span, half_span)


def _apply_square_axes(ax, x_axis, y_axis):
    xlim, ylim = _square_limits(x_axis, y_axis)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def _slm_axes(cfg):
    x = (np.arange(cfg.Nx) - cfg.Nx / 2) * cfg.spix / 1000
    y = (np.arange(cfg.Ny) - cfg.Ny / 2) * cfg.spix / 1000
    return x, y


def _focal_plane_axes_um(cfg):
    wavelength_um = cfg.wavelength_nm / 1e3
    focal_length_um = cfg.lens_focal_length_mm * 1e3
    Lx_um = cfg.Nx * cfg.spix
    Ly_um = cfg.Ny * cfg.spix
    delta_x_um = wavelength_um * focal_length_um / Lx_um
    delta_y_um = wavelength_um * focal_length_um / Ly_um
    x_um = (np.arange(cfg.NTx) - cfg.NTx / 2) * delta_x_um
    y_um = (np.arange(cfg.NTy) - cfg.NTy / 2) * delta_y_um
    return x_um, y_um


def _plot_center_profiles(ax, image, target, x_axis, y_axis, title, xlabel):
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    ax.plot(x_axis, target[cy, :], lw=2.0, color="#1f77b4", label="Target x-cut")
    ax.plot(x_axis, image[cy, :], lw=2.0, color="#d62728", label="Output x-cut")
    ax.plot(y_axis, target[:, cx], lw=1.8, ls="--", color="#2ca02c", label="Target y-cut")
    ax.plot(y_axis, image[:, cx], lw=1.8, ls="--", color="#9467bd", label="Output y-cut")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized intensity")
    ax.legend(frameon=True, fontsize=8, loc="best")


def _format_input_fit(input_beam_fit):
    if not input_beam_fit:
        return "Input Gaussian fit: unavailable"

    fit_x = input_beam_fit.get("x", {})
    fit_y = input_beam_fit.get("y", {})
    dx = fit_x.get("diameter_mm", np.nan)
    dy = fit_y.get("diameter_mm", np.nan)
    if not np.isfinite(dx) or not np.isfinite(dy):
        return "Input Gaussian fit: failed"
    return f"Input Gaussian fit: Dx={dx:.3f} mm, Dy={dy:.3f} mm"


def create_initial_figure(run_data, save_path=None, show=True):
    cfg = run_data["config"]
    input_beam = run_data["input_beam"]
    target_amp = run_data["target_amplitude"]
    weighting = run_data["weighting_mask"]
    ref_phase = run_data["reference_phase"]
    init_phase = run_data["initial_phase"]
    radius = run_data["plot_radius"]
    focal_x = run_data["focal_x_um"]
    focal_y = run_data["focal_y_um"]

    target_intensity = target_amp**2
    target_crop, bounds = _crop_roi(target_intensity, focal_x, focal_y, radius)

    slm_x, slm_y = _slm_axes(cfg)
    x_min, x_max, y_min, y_max = bounds
    crop_x = focal_x[x_min:x_max]
    crop_y = focal_y[y_min:y_max]

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    axs = fig.subplots(2, 3)
    fig.suptitle(
        f"Initial Setup Summary | {_format_input_fit(run_data.get('input_beam_fit'))}",
        fontsize=16,
        fontweight="bold",
    )

    im1 = axs[0, 0].imshow(
        input_beam,
        origin="lower",
        cmap="viridis",
        extent=_axis_extent_from_coords(slm_x, slm_y),
        aspect="equal",
    )
    axs[0, 0].set_title("Input Beam Amplitude")
    axs[0, 0].set_xlabel("SLM x (mm)")
    axs[0, 0].set_ylabel("SLM y (mm)")
    axs[0, 0].set_aspect("equal")
    fig.colorbar(im1, ax=axs[0, 0], shrink=0.84)

    im2 = axs[0, 1].imshow(
        target_intensity,
        origin="lower",
        cmap="inferno",
        extent=_axis_extent_from_coords(focal_x, focal_y),
        aspect="auto",
    )
    axs[0, 1].set_title("Target Intensity")
    axs[0, 1].set_xlabel("Focal-plane x (um)")
    axs[0, 1].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axs[0, 1], focal_x, focal_y)
    fig.colorbar(im2, ax=axs[0, 1], shrink=0.84)

    im3 = axs[0, 2].imshow(
        weighting,
        origin="lower",
        cmap="gray",
        extent=_axis_extent_from_coords(focal_x, focal_y),
        aspect="equal",
    )
    axs[0, 2].set_title("Weighting Mask")
    axs[0, 2].set_xlabel("Focal-plane x (um)")
    axs[0, 2].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axs[0, 2], focal_x, focal_y)
    fig.colorbar(im3, ax=axs[0, 2], shrink=0.84)

    im4 = axs[1, 0].imshow(
        ref_phase,
        origin="lower",
        cmap="twilight",
        extent=_axis_extent_from_coords(focal_x, focal_y),
        aspect="auto",
    )
    axs[1, 0].set_title("Target Phase")
    axs[1, 0].set_xlabel("Focal-plane x (um)")
    axs[1, 0].set_ylabel("Focal-plane y (um)")
    _apply_square_axes(axs[1, 0], focal_x, focal_y)
    fig.colorbar(im4, ax=axs[1, 0], shrink=0.84)

    im5 = axs[1, 1].imshow(
        init_phase,
        origin="lower",
        cmap="twilight",
        extent=_axis_extent_from_coords(slm_x, slm_y),
        aspect="auto",
    )
    axs[1, 1].set_title("Initial Hologram Phase Guess")
    axs[1, 1].set_xlabel("SLM x (mm)")
    axs[1, 1].set_ylabel("SLM y (mm)")
    axs[1, 1].set_aspect("equal")
    fig.colorbar(im5, ax=axs[1, 1], shrink=0.84)

    target_crop = _normalize_image(target_crop)
    _plot_center_profiles(
        axs[1, 2],
        target_crop,
        target_crop,
        crop_x,
        crop_y,
        "Target ROI Profiles",
        "Position (um)",
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def create_result_figure(run_data, save_path=None, show=True):
    cfg = run_data["config"]
    output_intensity = run_data["output_intensity"]
    target_intensity = run_data["target_amplitude"]**2
    output_phase = run_data["output_phase"]
    reference_phase = run_data["reference_phase"]
    loss_history = run_data["iteration_loss_history"] or run_data["loss_history"]
    metrics = run_data["metrics"]
    radius = run_data["plot_radius"]
    focal_x = run_data["focal_x_um"]
    focal_y = run_data["focal_y_um"]

    out_crop, bounds = _crop_roi(output_intensity, focal_x, focal_y, radius)
    tar_crop, _ = _crop_roi(target_intensity, focal_x, focal_y, radius)
    out_phase_crop, _ = _crop_roi(output_phase, focal_x, focal_y, radius)
    ref_phase_crop, _ = _crop_roi(reference_phase, focal_x, focal_y, radius)

    out_crop = _normalize_image(out_crop)
    tar_crop = _normalize_image(tar_crop)

    x_min, x_max, y_min, y_max = bounds
    crop_x = focal_x[x_min:x_max]
    crop_y = focal_y[y_min:y_max]
    crop_extent = _axis_extent_from_coords(crop_x, crop_y)

    fullres_phase = run_data["fullres_hologram_phase"]

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    plt.tight_layout()
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.05], height_ratios=[1, 1, 1])
    fig.suptitle("Optimization Result Summary", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(tar_crop, origin="lower", cmap="inferno", vmin=0, vmax=1, extent=crop_extent, aspect="auto")
    ax1.set_title("Target ROI Intensity")
    ax1.set_xlabel("Focal-plane x (um)")
    ax1.set_ylabel("Focal-plane y (um)")
    _apply_square_axes(ax1, crop_x, crop_y)
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(out_crop, origin="lower", cmap="inferno", vmin=0, vmax=1, extent=crop_extent, aspect="auto")
    ax2.set_title("Output ROI Intensity")
    ax2.set_xlabel("Focal-plane x (um)")
    ax2.set_ylabel("Focal-plane y (um)")
    _apply_square_axes(ax2, crop_x, crop_y)
    fig.colorbar(im2, ax=ax2, shrink=0.84)

    ax3 = fig.add_subplot(gs[0, 2:4])
    _plot_center_profiles(ax3, out_crop, tar_crop, crop_x, crop_y, "Target vs Output Profiles", "Position (um)")

    ax4 = fig.add_subplot(gs[1, 0:2])
    im4 = ax4.imshow(
        fullres_phase,
        origin="lower",
        cmap="twilight",
        aspect="auto",
    )
    ax4.set_title("Restored Full-resolution Hologram Phase")
    ax4.set_xlabel("Pixel x")
    ax4.set_ylabel("Pixel y")
    ax4.set_aspect("equal", adjustable="box")
    fig.colorbar(im4, ax=ax4, shrink=0.84)

    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(out_phase_crop, origin="lower", cmap="twilight", extent=crop_extent, aspect="auto")
    ax5.set_title("Far-field Phase in ROI")
    ax5.set_xlabel("Focal-plane x (um)")
    ax5.set_ylabel("Focal-plane y (um)")
    _apply_square_axes(ax5, crop_x, crop_y)
    fig.colorbar(im5, ax=ax5, shrink=0.84)

    ax6 = fig.add_subplot(gs[1, 3])
    im6 = ax6.imshow(ref_phase_crop, origin="lower", cmap="twilight", extent=crop_extent, aspect="auto")
    ax6.set_title("Target Phase Reference")
    ax6.set_xlabel("Focal-plane x (um)")
    ax6.set_ylabel("Focal-plane y (um)")
    _apply_square_axes(ax6, crop_x, crop_y)
    fig.colorbar(im6, ax=ax6, shrink=0.84)

    ax7 = fig.add_subplot(gs[2, 0])
    iter_axis = np.arange(1, len(loss_history) + 1)
    ax7.plot(iter_axis, loss_history, color="#d62728", lw=2.2, marker="o", ms=3.5, label="Loss")
    ax7.set_yscale("log")
    ax7.set_title("Loss History per Iteration")
    ax7.set_xlabel("Iteration")
    ax7.set_ylabel("Loss")
    ax7.legend(frameon=True)

    ax8 = fig.add_subplot(gs[2, 1:3], projection="3d")
    yy, xx = np.meshgrid(crop_y, crop_x, indexing="ij")
    stride_y = max(1, out_crop.shape[0] // 60)
    stride_x = max(1, out_crop.shape[1] // 60)
    surf = ax8.plot_surface(
        xx[::stride_y, ::stride_x],
        yy[::stride_y, ::stride_x],
        out_crop[::stride_y, ::stride_x],
        cmap="magma",
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )
    ax8.contour(xx, yy, out_crop, zdir="z", offset=0, cmap="magma", levels=12, linewidths=0.7)
    ax8.view_init(elev=33, azim=-128)
    ax8.set_title("3D Output Intensity Surface")
    ax8.set_xlabel("x (um)")
    ax8.set_ylabel("y (um)")
    ax8.set_zlabel("Normalized intensity")
    fig.colorbar(surf, ax=ax8, shrink=0.70, pad=0.08)

    ax9 = fig.add_subplot(gs[2, 3])
    input_fit_text = _format_input_fit(run_data.get("input_beam_fit"))
    summary_text = (
        f"Efficiency: {metrics['efficiency'] * 100:.2f}%\n"
        f"Fidelity: {metrics['fidelity']:.4f}\n"
        f"RMS Error: {metrics['rms_error'] * 100:.2f}%\n"
        f"Phase Error: {metrics['phase_error'] * 100:.2f}%\n"
        f"{input_fit_text}\n"
        f"Optimization time: {run_data['optimization_time_sec']:.2f} s\n"
        f"Total time: {run_data['total_time_sec']:.2f} s\n"
        f"Iterations: {run_data['optimizer_result'].nit}\n"
        f"Function evals: {run_data['optimizer_result'].nfev}\n"
        f"Device: {run_data['device']}\n"
        f"Mode: {cfg.target_mode}\n"
        f"Superpixel factor: {cfg.superpixel_factor}"
    )
    ax9.axis("off")
    ax9.text(
        0.02,
        0.98,
        summary_text,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F4F4", edgecolor="#CCCCCC"),
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

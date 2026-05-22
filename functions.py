import numpy as np
import scipy
import scipy.signal
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, convolve, gaussian_filter
from scipy.ndimage import zoom


def laser_gaussian(nx, ny, r0, sigmax, sigmay, A=1.0, plot=False):
    x = np.arange(nx) - nx/2
    y = np.arange(ny) - ny/2
    X, Y = np.meshgrid(x, y, indexing='xy')

    # Convert to the 1/e^2 Gaussian convention.
    sigmax_eff = np.sqrt(2) * sigmax
    sigmay_eff = np.sqrt(2) * sigmay

    Z = A * np.exp(-2 * (((X - r0[0]) / sigmax_eff)**2 +
                        ((Y - r0[1]) / sigmay_eff)**2))

    if plot:
        plt.figure(figsize=(6,5))
        plt.imshow(Z, origin='lower', cmap='viridis',
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.colorbar(label='Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gaussian Beam Intensity')
        plt.tight_layout()
        plt.show()

        # Central line cuts.
        cx = int(nx // 2 + r0[0])
        cy = int(ny // 2 + r0[1])

        plt.figure(figsize=(6,4))
        plt.plot(x, Z[cy, :], label='x cut')
        plt.plot(y, Z[:, cx], label='y cut')
        plt.xlabel('Position')
        plt.ylabel('Intensity')
        plt.legend()
        plt.title('Cross Sections')
        plt.grid()
        plt.tight_layout()
        plt.show()

    return Z


def _gaussian_1e2(x, offset, amplitude, center, radius):
    return offset + amplitude * np.exp(-2 * ((x - center) / radius) ** 2)


def _fit_gaussian_diameter_1d(axis_mm, profile):
    axis_mm = np.asarray(axis_mm, dtype=float)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(axis_mm) & np.isfinite(profile)
    axis_mm = axis_mm[finite]
    profile = profile[finite]

    if axis_mm.size < 5 or np.max(profile) <= 0:
        return {"diameter_mm": np.nan, "center_mm": np.nan, "radius_mm": np.nan, "success": False}

    baseline = np.percentile(profile, 5)
    shifted = np.clip(profile - baseline, 0.0, None)
    weight_sum = np.sum(shifted)
    if weight_sum <= 0:
        return {"diameter_mm": np.nan, "center_mm": np.nan, "radius_mm": np.nan, "success": False}

    center0 = np.sum(axis_mm * shifted) / weight_sum
    variance = np.sum(((axis_mm - center0) ** 2) * shifted) / weight_sum
    radius0 = max(np.sqrt(max(2 * variance, 0.0)), np.mean(np.diff(axis_mm)))
    amplitude0 = max(np.max(profile) - baseline, 1e-12)

    span = max(axis_mm[-1] - axis_mm[0], np.mean(np.diff(axis_mm)))
    lower = [0.0, 0.0, axis_mm[0], np.mean(np.diff(axis_mm))]
    upper = [np.max(profile), np.inf, axis_mm[-1], 2 * span]

    try:
        popt, _ = scipy.optimize.curve_fit(
            _gaussian_1e2,
            axis_mm,
            profile,
            p0=[max(baseline, 0.0), amplitude0, center0, radius0],
            bounds=(lower, upper),
            maxfev=10000,
        )
        radius = abs(popt[3])
        return {
            "diameter_mm": 2 * radius,
            "center_mm": popt[2],
            "radius_mm": radius,
            "offset": popt[0],
            "amplitude": popt[1],
            "success": True,
        }
    except (RuntimeError, ValueError):
        return {
            "diameter_mm": 2 * radius0,
            "center_mm": center0,
            "radius_mm": radius0,
            "offset": max(baseline, 0.0),
            "amplitude": amplitude0,
            "success": False,
        }


def fit_input_beam_gaussian_diameters(cfg, input_beam):
    x_mm = (np.arange(cfg.Nx) - cfg.Nx / 2) * cfg.spix / 1000
    y_mm = (np.arange(cfg.Ny) - cfg.Ny / 2) * cfg.spix / 1000
    intensity = np.abs(input_beam) ** 2
    profile_x = np.sum(intensity, axis=0)
    profile_y = np.sum(intensity, axis=1)

    fit_x = _fit_gaussian_diameter_1d(x_mm, profile_x)
    fit_y = _fit_gaussian_diameter_1d(y_mm, profile_y)
    return {
        "x": fit_x,
        "y": fit_y,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "axis_x_mm": x_mm,
        "axis_y_mm": y_mm,
    }

def expand_superpixel(data, factor=75):
    return np.kron(data, np.ones((factor, factor)))


def _resolve_h5_dataset(handle, dataset_path):
    if dataset_path in handle:
        return handle[dataset_path]

    if dataset_path.startswith("f.") and dataset_path[2:] in handle:
        return handle[dataset_path[2:]]

    available = []
    handle.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
    raise KeyError(f"Dataset '{dataset_path}' was not found. Available datasets: {available}")


def _resample_to_shape(data, target_shape):
    target_y, target_x = target_shape
    if data.shape == target_shape:
        return data

    zoom_y = target_y / data.shape[0]
    zoom_x = target_x / data.shape[1]
    return zoom(data, (zoom_y, zoom_x), order=1)


def load_measured_power_h5(
    h5_file,
    dataset_path="power",
    target_shape=None,
    power_is_intensity=True,
    smoothing_sigma=0.0,
    background_percentile=None,
):
    """
    Load an experimentally measured input-beam profile from an HDF5 power dataset.
    """

    with h5py.File(h5_file, "r") as f:
        power = np.asarray(_resolve_h5_dataset(f, dataset_path))

    power = np.squeeze(power).astype(float)
    power[np.isinf(power)] = np.nan

    if power.ndim > 2:
        if power.shape[-2:] == (0, 0):
            raise ValueError(f"Measured power dataset has invalid shape {power.shape}")
        profile_shape = power.shape[-2:]
        profiles = power.reshape(-1, *profile_shape)
        valid_counts = np.sum(~np.isnan(profiles), axis=0)
        summed = np.nansum(profiles, axis=0)
        power = np.divide(
            summed,
            valid_counts,
            out=np.full(profile_shape, np.nan, dtype=float),
            where=valid_counts > 0,
        )
    elif power.ndim != 2:
        raise ValueError(f"Measured power dataset must be 2D or stack of 2D profiles, got shape {power.shape}")

    power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

    power = np.clip(power, 0.0, None)

    if background_percentile is not None:
        background = np.percentile(power, background_percentile)
        power = np.clip(power - background, 0.0, None)

    if smoothing_sigma and smoothing_sigma > 0:
        power = gaussian_filter(power, sigma=smoothing_sigma)

    if np.max(power) <= 0:
        raise ValueError("Measured power dataset contains no positive values after preprocessing.")

    power = power / np.max(power)
    amplitude = np.sqrt(power) if power_is_intensity else power

    if target_shape is not None:
        amplitude = _resample_to_shape(amplitude, target_shape)

    power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
    amplitude = np.clip(amplitude, 0.0, None)
    if np.max(amplitude) <= 0:
        raise ValueError("Measured beam amplitude is zero after resampling.")

    print(f"Successfully loaded measured beam from {h5_file} dataset '{dataset_path}' with shape {amplitude.shape}")

    return amplitude / np.max(amplitude)


def ideal_line_shape_target(x_axis: np.ndarray, y_axis: np.ndarray, width_x_um: float, gaussian_diameter_y_um: float) -> np.ndarray:
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    top_hat_x = (np.abs(X) <= width_x_um / 2).astype(float)
    radius_y_um = gaussian_diameter_y_um / 2
    if radius_y_um <= 0:
        raise ValueError(f"gaussian_diameter_y_um must be positive, got {gaussian_diameter_y_um}.")
    gaussian_y = np.exp(-2 * (Y / radius_y_um) ** 2)
    return top_hat_x * gaussian_y


def ideal_rectangle_target(x_axis: np.ndarray, y_axis: np.ndarray, width_x_um: float, width_y_um: float) -> np.ndarray:
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    top_hat_x = np.abs(X) <= width_x_um / 2
    top_hat_y = np.abs(Y) <= width_y_um / 2
    return (top_hat_x & top_hat_y).astype(float)


def estimate_target_psf_sigma_um(cfg) -> tuple[float, float]:
    wavelength_um = cfg.wavelength_nm / 1e3
    focal_length_um = cfg.lens_focal_length_mm * 1e3
    beam_radius_x_um = cfg.beam_diameter_x_mm * 1e3 / 2
    beam_radius_y_um = cfg.beam_diameter_y_mm * 1e3 / 2
    if beam_radius_x_um <= 0 or beam_radius_y_um <= 0:
        raise ValueError(
            "Beam diameters must be positive to estimate target PSF. "
            f"beam_diameter_x_mm={cfg.beam_diameter_x_mm}, beam_diameter_y_mm={cfg.beam_diameter_y_mm}."
        )

    diffraction_sigma_x_um = wavelength_um * focal_length_um / (np.sqrt(2) * np.pi * beam_radius_x_um)
    diffraction_sigma_y_um = wavelength_um * focal_length_um / (np.sqrt(2) * np.pi * beam_radius_y_um)
    camera_sigma_um = cfg.camera_pixel_size_um / np.sqrt(12)
    sampling_delta_x_um, sampling_delta_y_um = get_focal_plane_sampling_um(cfg)
    sampling_sigma_x_um = sampling_delta_x_um / np.sqrt(12)
    sampling_sigma_y_um = sampling_delta_y_um / np.sqrt(12)

    sigma_x_um = np.sqrt(diffraction_sigma_x_um**2 + camera_sigma_um**2 + sampling_sigma_x_um**2)
    sigma_y_um = np.sqrt(diffraction_sigma_y_um**2 + camera_sigma_um**2 + sampling_sigma_y_um**2)
    return float(sigma_x_um), float(sigma_y_um)


def resolve_target_psf_sigma_um(cfg) -> tuple[float, float]:
    estimated_sigma_x_um, estimated_sigma_y_um = estimate_target_psf_sigma_um(cfg)
    sigma_x_um = estimated_sigma_x_um if cfg.target_psf_sigma_x_um is None else float(cfg.target_psf_sigma_x_um)
    sigma_y_um = estimated_sigma_y_um if cfg.target_psf_sigma_y_um is None else float(cfg.target_psf_sigma_y_um)
    if sigma_x_um < 0 or sigma_y_um < 0:
        raise ValueError(f"Target PSF sigma values must be non-negative, got sigma_x={sigma_x_um}, sigma_y={sigma_y_um}.")
    return sigma_x_um, sigma_y_um


def build_gaussian_psf_kernel(
    delta_x_um: float,
    delta_y_um: float,
    sigma_x_um: float,
    sigma_y_um: float,
    truncate: float,
) -> np.ndarray:
    if sigma_x_um < 0 or sigma_y_um < 0:
        raise ValueError(f"PSF sigma values must be non-negative, got sigma_x={sigma_x_um}, sigma_y={sigma_y_um}.")
    if truncate <= 0:
        raise ValueError(f"target_psf_truncate must be positive, got {truncate}.")
    if sigma_x_um == 0 and sigma_y_um == 0:
        return np.array([[1.0]], dtype=float)

    sigma_x_px = max(sigma_x_um / delta_x_um, np.finfo(float).eps)
    sigma_y_px = max(sigma_y_um / delta_y_um, np.finfo(float).eps)
    radius_x = max(1, int(np.ceil(truncate * sigma_x_px)))
    radius_y = max(1, int(np.ceil(truncate * sigma_y_px)))
    x = np.arange(-radius_x, radius_x + 1, dtype=float)
    y = np.arange(-radius_y, radius_y + 1, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    kernel = np.exp(-0.5 * ((X / sigma_x_px) ** 2 + (Y / sigma_y_px) ** 2))
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0:
        raise ValueError("Gaussian PSF kernel has zero total weight.")
    return kernel / kernel_sum


def convolve_target_with_psf(target: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
    target_array = np.asarray(target, dtype=float)
    kernel_array = np.asarray(psf_kernel, dtype=float)
    if target_array.ndim != 2:
        raise ValueError(f"Target must be 2D, got shape {target_array.shape}.")
    if kernel_array.ndim != 2:
        raise ValueError(f"PSF kernel must be 2D, got shape {kernel_array.shape}.")
    return scipy.signal.fftconvolve(target_array, kernel_array, mode="same")


def smooth_with_gaussian(z, sigma):
    return gaussian_filter(z, sigma=sigma)


def smooth_with_convolution(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    data = np.asarray(z, dtype=float)
    kernel_array = np.asarray(kernel, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"Convolution smoothing expects a 2D array, got shape {data.shape}.")
    if kernel_array.ndim != 2 or kernel_array.size == 0:
        raise ValueError(f"Convolution kernel must be a non-empty 2D array, got shape {kernel_array.shape}.")
    if not np.all(np.isfinite(kernel_array)):
        raise ValueError("Convolution kernel contains non-finite values.")

    kernel_sum = float(np.sum(kernel_array))
    if abs(kernel_sum) <= np.finfo(float).eps:
        raise ValueError("Convolution kernel sum must be non-zero for normalized smoothing.")

    normalized_kernel = kernel_array / kernel_sum
    return convolve(data, normalized_kernel, mode="nearest")


def smooth_with_box_convolution(z: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd so the kernel has a center pixel, got {kernel_size}.")

    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    return smooth_with_convolution(z, kernel)


def weighting_value(M, p, v=0):
    z = np.ones_like(M)
    z[np.abs(M) < p * M.max()] = v
    return z


def build_mask_from_target(target, margin_pixels=0, threshold=1e-2, background=0.0):
    target = np.asarray(target)
    if target.size == 0 or np.max(np.abs(target)) <= 0:
        raise ValueError("Cannot build a mask from an empty or zero target.")

    core = np.abs(target) > threshold * np.max(np.abs(target))
    if margin_pixels > 0:
        structure = np.ones((2 * int(margin_pixels) + 1, 2 * int(margin_pixels) + 1), dtype=bool)
        core = binary_dilation(core, structure=structure)

    mask = np.full(target.shape, background, dtype=float)
    mask[core] = 1.0
    return mask


def phase_guess_2d(nx, ny, D, asp, R, ang, B):
    x = np.arange(nx) - nx/2
    y = np.arange(ny) - ny/2
    X, Y = np.meshgrid(x, y, indexing='xy')

    KL = D * (X*np.cos(ang) + Y*np.sin(ang))
    KQ = 3 * R * (asp*(X**2) + (1-asp)*(Y**2))
    KC = B * np.sqrt(X**2 + Y**2)

    z = KC + KQ + KL
    return z.reshape(-1)

def phase_guess_2d_unreshaped(nx, ny, D, asp, R, ang, B, superpixel_factor):
    x = np.arange(nx) - nx/2
    y = np.arange(ny) - ny/2
    X, Y = np.meshgrid(x, y, indexing='xy')

    KL = D * (X*np.cos(ang) + Y*np.sin(ang))
    KQ = 3 * R * (asp*(X**2) + (1-asp)*(Y**2))
    KC = B * np.sqrt(X**2 + Y**2)

    z = KC + KQ + KL
    z = np.mod(expand_superpixel(z, superpixel_factor), 2 * np.pi)
    return z

def phase_gradient(nx, ny, kx, ky):
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    return kx * X + ky * Y


def get_focal_plane_sampling_um(cfg):
    wavelength_um = cfg.wavelength_nm / 1e3
    focal_length_um = cfg.lens_focal_length_mm * 1e3
    Lx_um = cfg.Nx * cfg.spix
    Ly_um = cfg.Ny * cfg.spix
    delta_x_um = wavelength_um * focal_length_um / Lx_um
    delta_y_um = wavelength_um * focal_length_um / Ly_um
    return delta_x_um, delta_y_um


def get_focal_plane_axes_um(cfg):
    delta_x_um, delta_y_um = get_focal_plane_sampling_um(cfg)
    x_um = (np.arange(cfg.NTx) - cfg.NTx / 2) * delta_x_um
    y_um = (np.arange(cfg.NTy) - cfg.NTy / 2) * delta_y_um
    return x_um, y_um


def build_target(cfg):
    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)
    delta_x_um, delta_y_um = get_focal_plane_sampling_um(cfg)

    if cfg.target_mode == "line_shape":
        ideal_target = ideal_line_shape_target(
            focal_x_um,
            focal_y_um,
            cfg.line_width_x_um,
            cfg.line_width_y_um,
        )
    elif cfg.target_mode == "rectangle":
        ideal_target = ideal_rectangle_target(
            focal_x_um,
            focal_y_um,
            cfg.rect_width_x_um,
            cfg.rect_width_y_um,
        )
    else:
        raise ValueError(f"Unsupported target_mode: {cfg.target_mode}")

    psf_sigma_x_um, psf_sigma_y_um = resolve_target_psf_sigma_um(cfg)
    psf_kernel = build_gaussian_psf_kernel(
        delta_x_um,
        delta_y_um,
        psf_sigma_x_um,
        psf_sigma_y_um,
        cfg.target_psf_truncate,
    )
    target = convolve_target_with_psf(ideal_target, psf_kernel)
    return np.clip(target, 0.0, None)


def build_weighting_mask(cfg, target=None):
    if target is not None:
        delta_x_um, delta_y_um = get_focal_plane_sampling_um(cfg)
        margin_pixels = int(np.ceil(cfg.mask_margin_um / min(delta_x_um, delta_y_um)))
        return build_mask_from_target(
            target,
            margin_pixels=margin_pixels,
            threshold=cfg.weighting_threshold,
            background=cfg.weighting_background,
        )

    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)
    X, Y = np.meshgrid(focal_x_um, focal_y_um, indexing="xy")

    if cfg.mask_shape == "circle" or (cfg.mask_shape == "auto" and cfg.target_mode == "line_shape"):
        radius_um = max(cfg.line_width_x_um, cfg.line_width_y_um) / 2 + cfg.mask_margin_um
        mask = (X**2 + Y**2) <= radius_um**2
    elif cfg.mask_shape == "rectangle" or (cfg.mask_shape == "auto" and cfg.target_mode == "rectangle"):
        mask = (
            (np.abs(X) <= cfg.rect_width_x_um / 2 + cfg.mask_margin_um)
            & (np.abs(Y) <= cfg.rect_width_y_um / 2 + cfg.mask_margin_um)
        )
    else:
        raise ValueError(f"Unsupported mask_shape: {cfg.mask_shape}")

    weighting = np.zeros((cfg.NTy, cfg.NTx))
    weighting[mask] = 1.0
    return weighting_value(weighting, cfg.weighting_threshold, cfg.weighting_background)


def get_plot_radius(cfg):
    if cfg.target_mode == "line_shape":
        base_radius_um = max(cfg.line_width_x_um, cfg.line_width_y_um) / 2 + cfg.mask_margin_um
    else:
        base_radius_um = max(cfg.rect_width_x_um, cfg.rect_width_y_um) / 2 + cfg.mask_margin_um
    return base_radius_um * cfg.output_crop_factor


def downsample_average(arr, factor):
    arr = np.asarray(arr)
    h, w = arr.shape
    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor
    arr_trim = arr[:h_trim, :w_trim]
    return arr_trim.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))


def build_input_beam(cfg):
    if cfg.input_beam_source == "gaussian":
        beam_radius_x = cfg.beam_diameter_x_mm * 1000 / 2
        beam_radius_y = cfg.beam_diameter_y_mm * 1000 / 2
        input_beam = laser_gaussian(cfg.Nx, cfg.Ny, (0, 0), beam_radius_x / cfg.spix, beam_radius_y / cfg.spix)
        return fit_input_beam_gaussian_diameters(cfg, input_beam), input_beam

    if cfg.input_beam_source == "measured_h5":
        if not cfg.input_beam_h5_path:
            raise ValueError("input_beam_h5_path must be set when input_beam_source='measured_h5'.")
        input_beam = load_measured_power_h5(
            cfg.input_beam_h5_path,
            dataset_path=cfg.input_beam_h5_dataset,
            target_shape=(cfg.Ny, cfg.Nx),
            power_is_intensity=cfg.measured_power_is_intensity,
            smoothing_sigma=cfg.measured_beam_smoothing_sigma,
            background_percentile=cfg.measured_beam_background_percentile,
        )
        return fit_input_beam_gaussian_diameters(cfg, input_beam), input_beam

    if cfg.input_beam_source == "slmsuite_h5":
        return fitting_input_beam(cfg)

    raise ValueError(f"Unsupported input_beam_source: {cfg.input_beam_source}")


def fitting_input_beam(cfg):
    from slmsuite.hardware.cameras.simulated import SimulatedCamera
    from slmsuite.hardware.cameraslms import FourierSLM
    from slmsuite.hardware.slms.simulated import SimulatedSLM

    slm_size = (cfg.full_slm_Nx, cfg.full_slm_Ny)
    cam_size = (cfg.full_cam_Nx, cfg.full_cam_Ny)
    wav_um = cfg.wavelength_nm / 1000

    slm = SimulatedSLM(slm_size, pitch_um=(cfg.base_spix_um, cfg.base_spix_um), wav_um=wav_um)
    camera = SimulatedCamera(slm, resolution=cam_size)
    fs = FourierSLM(camera, slm)

    fs.load_calibration(cfg.input_beam_h5_path)
    calibration_results = fs.wavefront_calibration_superpixel_process(
        plot=False,
        r2_threshold=.5,
        remove_background=True,
        apply=True,
    )
    input_beam = downsample_average(calibration_results["amplitude"], factor=cfg.superpixel_factor)
    gaussian_fittings = fit_input_beam_gaussian_diameters(cfg, input_beam)

    wx = gaussian_fittings["x"]["radius_mm"]
    wy = gaussian_fittings["y"]["radius_mm"]
    cx = gaussian_fittings["x"]["center_mm"]
    cy = gaussian_fittings["y"]["center_mm"]
    x_mm = (np.arange(cfg.Nx) - cfg.Nx / 2) * cfg.spix / 1000
    y_mm = (np.arange(cfg.Ny) - cfg.Ny / 2) * cfg.spix / 1000
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
    fitted_beam = np.exp(-(((X_mm - cx) / wx) ** 2 + ((Y_mm - cy) / wy) ** 2))

    return gaussian_fittings, fitted_beam
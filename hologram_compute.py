from time import perf_counter

import numpy as np
import scipy.optimize
import torch

from benchmark import compute_benchmarks
from functions import (
    build_input_beam,
    build_target,
    build_weighting_mask,
    expand_superpixel,
    get_focal_plane_axes_um,
    get_plot_radius,
    phase_gradient,
    phase_guess_2d,
)
from logger import Logger


def cg_optimize(cfg):
    cfg.update_derived()
    total_start = perf_counter()

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Nx, Ny = cfg.Nx, cfg.Ny
    NTx, NTy = cfg.NTx, cfg.NTy
    logger = Logger()

    input_beam_fit, L_np = build_input_beam(cfg)
    L_np *= np.sqrt(cfg.input_power_norm / np.sum(L_np**2))

    Ta_np = build_target(cfg)
    P_np = phase_gradient(NTx, NTy, cfg.kx, cfg.ky)
    Wcg_np = build_weighting_mask(cfg)

    Ta_np *= Wcg_np
    Ta_np *= np.sqrt(np.sum(L_np**2) / np.sum(Ta_np**2))

    init_phi_np = phase_guess_2d(
        Nx,
        Ny,
        cfg.init_phase_d,
        cfg.init_phase_asp,
        cfg.curv / 1000,
        cfg.init_phase_ang,
        cfg.init_phase_b,
    )

    L = torch.tensor(L_np, dtype=dtype, device=device)
    Ta = torch.tensor(Ta_np, dtype=dtype, device=device)
    P = torch.tensor(P_np, dtype=dtype, device=device)
    Wcg = torch.tensor(Wcg_np, dtype=dtype, device=device)

    A0 = 1.0 / NTx
    x0 = NTx // 2 - Nx // 2
    y0 = NTy // 2 - Ny // 2

    def cost(phi_1d):
        phi = torch.tensor(phi_1d, dtype=dtype, device=device, requires_grad=True)
        phi2d = phi.view(Ny, Nx)

        E_slm = A0 * L * torch.exp(1j * phi2d)

        pad = torch.zeros((NTy, NTx), dtype=torch.complex128, device=device)
        pad[y0:y0 + Ny, x0:x0 + Nx] = E_slm

        E_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pad)))

        amp = torch.abs(E_out)
        ph = torch.angle(E_out)

        overlap = torch.sum(Ta * amp * Wcg * torch.cos(ph - P))
        overlap /= torch.sqrt(torch.sum(Ta**2) * torch.sum((amp * Wcg)**2))

        # add the efficiency term
        efficiency = torch.sum(amp * Wcg) / torch.sum(amp**2)

        loss = (10**cfg.C1) * ((1 - overlap)**2 - cfg.C2 * (efficiency - 1))
        loss.backward()
        logger.log_evaluation(loss.item())

        return loss.item(), phi.grad.cpu().numpy()

    def callback(xk):
        logger.log_iteration(xk)

    optimization_start = perf_counter()
    res = scipy.optimize.minimize(
        cost,
        init_phi_np,
        method=cfg.optimizer_method,
        jac=True,
        callback=callback,
        options={"maxiter": cfg.optimizer_maxiter, "disp": cfg.optimizer_disp},
    )
    optimization_time_sec = perf_counter() - optimization_start

    final_phi = torch.tensor(res.x, dtype=dtype, device=device).view(Ny, Nx)

    E_slm = A0 * L * torch.exp(1j * final_phi)
    pad = torch.zeros((NTy, NTx), dtype=torch.complex128, device=device)
    pad[y0:y0 + Ny, x0:x0 + Nx] = E_slm

    E_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pad))).cpu().numpy()

    eff, fid, rms, ph_err, I_out, Phase_out = compute_benchmarks(E_out, Ta_np, P_np, Wcg_np)
    total_time_sec = perf_counter() - total_start

    final_phase_wrapped = np.mod(res.x.reshape(Ny, Nx), 2 * np.pi)
    fullres_phase = np.mod(expand_superpixel(final_phase_wrapped, cfg.superpixel_factor), 2 * np.pi)
    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)

    return {
        "config": cfg.clone(),
        "metrics": {
            "efficiency": eff,
            "fidelity": fid,
            "rms_error": rms,
            "phase_error": ph_err,
        },
        "efficiency": eff,
        "fidelity": fid,
        "rms_error": rms,
        "phase_error": ph_err,
        "optimizer_result": res,
        "final_phase": final_phase_wrapped,
        "fullres_hologram_phase": fullres_phase,
        "initial_phase": np.mod(init_phi_np.reshape(Ny, Nx), 2 * np.pi),
        "input_beam": L_np,
        "input_beam_fit": input_beam_fit,
        "target_amplitude": Ta_np,
        "reference_phase": P_np,
        "weighting_mask": Wcg_np,
        "field_output": E_out,
        "output_intensity": I_out,
        "output_phase": Phase_out,
        "plot_radius": get_plot_radius(cfg),
        "focal_x_um": focal_x_um,
        "focal_y_um": focal_y_um,
        "loss_history": logger.eval_history,
        "iteration_loss_history": logger.iter_history,
        "optimization_time_sec": optimization_time_sec,
        "total_time_sec": total_time_sec,
        "device": str(device),
    }

def cg_optimize_with_target(cfg, Ta_np, last_opt_result, optimize_phase=False):
    cfg.update_derived()
    total_start = perf_counter()

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Nx, Ny = cfg.Nx, cfg.Ny
    NTx, NTy = cfg.NTx, cfg.NTy
    logger = Logger()

    input_beam_fit, L_np = build_input_beam(cfg)
    L_np *= np.sqrt(cfg.input_power_norm / np.sum(L_np**2))

    # Ta_np is passed as an argument this time
    P_np = phase_gradient(NTx, NTy, cfg.kx, cfg.ky)
    Wcg_np = build_weighting_mask(cfg)

    Ta_np *= Wcg_np
    Ta_np *= np.sqrt(np.sum(L_np**2) / np.sum(Ta_np**2))

    init_phi_np = last_opt_result["final_phase"].reshape(-1)

    L = torch.tensor(L_np, dtype=dtype, device=device)
    Ta = torch.tensor(Ta_np, dtype=dtype, device=device)
    P = torch.tensor(P_np, dtype=dtype, device=device)
    Wcg = torch.tensor(Wcg_np, dtype=dtype, device=device)

    A0 = 1.0 / NTx
    x0 = NTx // 2 - Nx // 2
    y0 = NTy // 2 - Ny // 2

    def cost(phi_1d):
        phi = torch.tensor(phi_1d, dtype=dtype, device=device, requires_grad=True)
        phi2d = phi.view(Ny, Nx)

        E_slm = A0 * L * torch.exp(1j * phi2d)

        pad = torch.zeros((NTy, NTx), dtype=torch.complex128, device=device)
        pad[y0:y0 + Ny, x0:x0 + Nx] = E_slm

        E_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pad)))

        amp = torch.abs(E_out)
        ph = torch.angle(E_out)

        if optimize_phase:
            overlap = torch.sum(Ta * amp * Wcg * torch.cos(ph - P))
        else:
            overlap = torch.sum(Ta * amp * Wcg)
        overlap /= torch.sqrt(torch.sum(Ta**2) * torch.sum((amp * Wcg)**2))

        loss = (10**cfg.C1) * (1 - overlap)**2
        loss.backward()
        logger.log_evaluation(loss.item())

        return loss.item(), phi.grad.cpu().numpy()

    def callback(xk):
        logger.log_iteration(xk)

    optimization_start = perf_counter()
    res = scipy.optimize.minimize(
        cost,
        init_phi_np,
        method=cfg.optimizer_method,
        jac=True,
        callback=callback,
        options={"maxiter": cfg.optimizer_maxiter, "disp": cfg.optimizer_disp},
    )
    optimization_time_sec = perf_counter() - optimization_start

    final_phi = torch.tensor(res.x, dtype=dtype, device=device).view(Ny, Nx)

    E_slm = A0 * L * torch.exp(1j * final_phi)
    pad = torch.zeros((NTy, NTx), dtype=torch.complex128, device=device)
    pad[y0:y0 + Ny, x0:x0 + Nx] = E_slm

    E_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(pad))).cpu().numpy()

    eff, fid, rms, ph_err, I_out, Phase_out = compute_benchmarks(E_out, Ta_np, P_np, Wcg_np)
    total_time_sec = perf_counter() - total_start

    final_phase_wrapped = np.mod(res.x.reshape(Ny, Nx), 2 * np.pi)
    fullres_phase = np.mod(expand_superpixel(final_phase_wrapped, cfg.superpixel_factor), 2 * np.pi)
    focal_x_um, focal_y_um = get_focal_plane_axes_um(cfg)

    return {
        "config": cfg.clone(),
        "metrics": {
            "efficiency": eff,
            "fidelity": fid,
            "rms_error": rms,
            "phase_error": ph_err,
        },
        "efficiency": eff,
        "fidelity": fid,
        "rms_error": rms,
        "phase_error": ph_err,
        "optimizer_result": res,
        "final_phase": final_phase_wrapped,
        "fullres_hologram_phase": fullres_phase,
        "initial_phase": np.mod(init_phi_np.reshape(Ny, Nx), 2 * np.pi),
        "input_beam": L_np,
        "input_beam_fit": input_beam_fit,
        "target_amplitude": Ta_np,
        "reference_phase": P_np,
        "weighting_mask": Wcg_np,
        "field_output": E_out,
        "output_intensity": I_out,
        "output_phase": Phase_out,
        "plot_radius": get_plot_radius(cfg),
        "focal_x_um": focal_x_um,
        "focal_y_um": focal_y_um,
        "loss_history": logger.eval_history,
        "iteration_loss_history": logger.iter_history,
        "optimization_time_sec": optimization_time_sec,
        "total_time_sec": total_time_sec,
        "device": str(device),
        "optimize_phase": optimize_phase,
    }


def _normalize_feedback_array(data, target_power, name):
    array = np.asarray(data, dtype=float).copy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, 0.0, None)
    power = float(np.sum(array**2))
    if power <= 0:
        raise ValueError(f"{name} contains no positive power and cannot be normalized.")
    return array * np.sqrt(target_power / power)


def _feedback_signal_mask(measured_amplitude):
    measured = np.asarray(measured_amplitude, dtype=float)
    mask = measured > 0
    if not np.any(mask):
        raise ValueError("Feedback measurement has no active pixels. Check get_image_func and the fitted support mask.")
    return mask.astype(float)


def _feedback_error_metrics(target, measured, signal_mask):
    active = signal_mask > 0
    residual = np.asarray(target, dtype=float)[active] - np.asarray(measured, dtype=float)[active]
    rmse = float(np.sqrt(np.mean(residual**2)))
    peak = float(np.max(np.asarray(target, dtype=float)[active]))
    psnr = np.inf if rmse <= 0 else float(20 * np.log10(max(peak, np.finfo(float).eps) / rmse))
    return rmse, psnr


def _normalize_optimize_axis(optimize_axis):
    if isinstance(optimize_axis, str):
        axes = [optimize_axis]
    else:
        axes = list(optimize_axis)
    axes = [axis.lower() for axis in axes]
    invalid_axes = sorted(set(axes) - {"x", "y"})
    if invalid_axes:
        raise ValueError(f"optimize_axis only supports 'x' and 'y', got {invalid_axes}.")
    if not axes:
        raise ValueError("optimize_axis must contain at least one axis.")
    return tuple(dict.fromkeys(axes))


def _center_profiles(data):
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Center profile extraction expects a 2D array, got shape {array.shape}.")
    center_y = array.shape[0] // 2
    center_x = array.shape[1] // 2
    return array[center_y, :].copy(), array[:, center_x].copy()


def _build_axis_feedback_target(T0, Ti, M, signal_mask, optimize_axis, alpha):
    axes = _normalize_optimize_axis(optimize_axis)
    T0_x, T0_y = _center_profiles(T0)
    Ti_x, Ti_y = _center_profiles(Ti)
    M_x, M_y = _center_profiles(M)

    next_x = Ti_x + float(alpha) * (T0_x - M_x) if "x" in axes else T0_x
    next_y = Ti_y + float(alpha) * (T0_y - M_y) if "y" in axes else T0_y
    next_x = np.clip(next_x, 0.0, None)
    next_y = np.clip(next_y, 0.0, None)
    next_x = np.where(Ti_x > 0, next_x, 0)
    next_y = np.where(Ti_y > 0, next_y, 0)

    if np.max(next_x) <= 0 or np.max(next_y) <= 0:
        raise ValueError("Axis feedback produced an empty target profile.")

    next_x /= np.max(next_x)
    next_y /= np.max(next_y)
    return np.outer(next_y, next_x) * signal_mask


def cg_optimize_optical_feedback(
    cfg,
    get_image_func,
    last_opt_result,
    smoothing_func=None,
    alpha=1.0,
    optimize_phase=False,
    optimize_axis=("x",),
    whole_plane_optimize=False,
):
    cfg.update_derived()

    original_target = np.asarray(
        last_opt_result.get("feedback_original_target", last_opt_result["target_amplitude"]),
        dtype=float,
    ).copy()
    current_target = np.asarray(last_opt_result["target_amplitude"], dtype=float).copy()
    measured_raw = np.asarray(get_image_func(), dtype=float)

    if measured_raw.shape != current_target.shape:
        raise ValueError(
            "Feedback measurement shape does not match the current target. "
            f"measurement_shape={measured_raw.shape}, target_shape={current_target.shape}."
        )

    signal_mask = _feedback_signal_mask(measured_raw)
    target_power = float(cfg.input_power_norm)
    T0 = _normalize_feedback_array(original_target * signal_mask, target_power, "Original feedback target")
    Ti = _normalize_feedback_array(current_target * signal_mask, target_power, "Current feedback target")
    M = _normalize_feedback_array(measured_raw * signal_mask, target_power, "Measured feedback image")

    if whole_plane_optimize:
        D = float(alpha) * (T0 - M)
        next_target = signal_mask * (Ti + D)
    else:
        next_target = _build_axis_feedback_target(T0, Ti, M, signal_mask, optimize_axis, alpha)
        D = next_target - Ti

    if smoothing_func is not None:
        D = np.asarray(smoothing_func(D), dtype=float)
        if D.shape != T0.shape:
            raise ValueError(f"smoothing_func changed D shape from {T0.shape} to {D.shape}.")
        D *= signal_mask
        next_target = signal_mask * (Ti + D)

    next_target = np.clip(next_target, 0.0, None)
    next_target = _normalize_feedback_array(next_target, target_power, "Next feedback target")

    rmse, psnr = _feedback_error_metrics(T0, M, signal_mask)
    cg_opt_result = cg_optimize_with_target(
        cfg,
        next_target.copy(),
        last_opt_result,
        optimize_phase=optimize_phase,
    )

    axes = _normalize_optimize_axis(optimize_axis)
    history = list(last_opt_result.get("feedback_history", []))
    history.append(
        {
            "alpha": float(alpha),
            "rmse": rmse,
            "psnr": psnr,
            "optimize_phase": bool(optimize_phase),
            "optimize_axis": axes,
            "whole_plane_optimize": bool(whole_plane_optimize),
        }
    )
    cg_opt_result["feedback_original_target"] = T0
    cg_opt_result["feedback_current_target"] = Ti
    cg_opt_result["feedback_measured"] = M
    cg_opt_result["feedback_discrepancy"] = D
    cg_opt_result["feedback_next_target_input"] = next_target
    cg_opt_result["feedback_signal_mask"] = signal_mask
    cg_opt_result["feedback_alpha"] = float(alpha)
    cg_opt_result["feedback_optimize_phase"] = bool(optimize_phase)
    cg_opt_result["feedback_optimize_axis"] = axes
    cg_opt_result["feedback_whole_plane_optimize"] = bool(whole_plane_optimize)
    cg_opt_result["feedback_rmse"] = rmse
    cg_opt_result["feedback_psnr"] = psnr
    cg_opt_result["feedback_history"] = history
    return cg_opt_result

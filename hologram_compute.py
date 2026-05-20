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
    }

def cg_optimize_with_target(cfg, Ta_np):
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
    }

def cg_optimize_optical_feedback(cfg, get_image_func, smoothing_func=None):
    # this function is used for optical feedback, the main idea is that 
    # we observed some other pattern I, with the pre-defined target T, 
    # then the new target is replaced into T' = T + D, where D = T - I, and sometimes we should do a smoothing on D
    # now we just ignore, this function only runs one iteration
    cfg.update_derived()
    Ta_np = build_target(cfg)
    I_observed = get_image_func()
    # normalize both to be fair
    Ta_np *= np.sqrt(cfg.input_power_norm / np.sum(Ta_np**2))
    I_observed *= np.sqrt(cfg.input_power_norm / np.sum(I_observed**2))
    D = Ta_np - I_observed
    if smoothing_func is not None:
        D = smoothing_func(D)
    cg_opt_result = cg_optimize_with_target(cfg, Ta_np + D)
    return cg_opt_result
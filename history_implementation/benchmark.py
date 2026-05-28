import torch
import numpy as np


def compute_benchmarks(E_out, Ta_np, P_np, Wcg_np, rel_eps=1e-6):

    I_out_np = np.abs(E_out)**2
    Phase_out_np = np.angle(E_out)

    # Weighted output and target intensities inside the ROI.
    I_out_w = I_out_np * Wcg_np
    I_tar_w = (Ta_np**2) * Wcg_np

    # Efficiency.
    efficiency = np.sum(I_out_w) / (np.sum(I_out_np) + 1e-12)

    # Fidelity.
    E_tar_w = (Ta_np * np.exp(1j * P_np)) * Wcg_np
    E_out_w = E_out * Wcg_np

    overlap_num = np.abs(np.sum(E_tar_w.conj() * E_out_w))**2
    overlap_den = np.sum(np.abs(E_tar_w)**2) * np.sum(np.abs(E_out_w)**2)
    fidelity = overlap_num / (overlap_den + 1e-12)

    # Normalize intensities only inside the ROI.
    mask = Wcg_np > 0

    I_out_norm = np.zeros_like(I_out_np)
    I_tar_norm = np.zeros_like(I_tar_w)

    I_out_norm[mask] = I_out_w[mask] / (np.sum(I_out_w[mask]) + 1e-12)
    I_tar_norm[mask] = I_tar_w[mask] / (np.sum(I_tar_w[mask]) + 1e-12)

    # Relative RMS error.
    valid = (mask) & (I_tar_norm > rel_eps)

    rel_err = np.zeros_like(I_out_np)
    rel_err[valid] = ((I_out_norm[valid] - I_tar_norm[valid]) / I_tar_norm[valid])**2

    rms_error = np.sqrt(np.sum(rel_err[valid]) / (np.sum(valid) + 1e-12))

    # Wrapped phase error.
    phase_diff = np.angle(np.exp(1j * (Phase_out_np - P_np)))

    phase_error = np.sqrt(
        np.sum((phase_diff**2) * Wcg_np) / (np.sum(Wcg_np) + 1e-12)
    )

    return efficiency, fidelity, rms_error, phase_error, I_out_np, Phase_out_np

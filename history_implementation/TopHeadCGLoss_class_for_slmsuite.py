import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import build_target, build_weighting_mask, phase_gradient


class TopHeadCGLoss(nn.Module):
    """Slmsuite-compatible CG loss for top-head beam shaping."""

    def __init__(self, cfg, target=None, include_phase=False):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.update_derived()
        self.use_passed_target = target is not None
        self.include_phase = include_phase
        self.target_shape = (self.cfg.NTy, self.cfg.NTx)

        if target is None:
            target_amplitude = build_target(self.cfg)
        else:
            target_amplitude = torch.as_tensor(target)
            if torch.is_complex(target_amplitude):
                target_amplitude = torch.abs(target_amplitude)
            target_amplitude = self._resize_real(
                target_amplitude.to(dtype=torch.float64),
                self.target_shape,
            )
            target_amplitude = target_amplitude.detach().cpu().numpy()

        weighting_mask = build_weighting_mask(self.cfg, target_amplitude)
        phase_reference = phase_gradient(self.cfg.NTx, self.cfg.NTy, self.cfg.kx, self.cfg.ky)

        self.register_buffer(
            "target_amplitude",
            torch.as_tensor(target_amplitude, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "phase_reference",
            torch.as_tensor(phase_reference, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "weighting_mask",
            torch.as_tensor(weighting_mask, dtype=torch.float64),
            persistent=False,
        )

    def _prepare_buffer(self, tensor, farfield):
        return tensor.to(device=farfield.device, dtype=torch.float64)

    def _resize_real(self, tensor, shape):
        if tuple(tensor.shape) == tuple(shape):
            return tensor
        return F.interpolate(tensor[None, None, :, :], size=shape, mode="area")[0, 0]

    def _target_from_argument(self, target, farfield):
        target_tensor = torch.as_tensor(target, device=farfield.device)
        if torch.is_complex(target_tensor):
            target_tensor = torch.abs(target_tensor)
        return target_tensor.to(dtype=torch.float64)

    def forward(self, farfield, target=None):
        farfield = torch.as_tensor(farfield)
        target_shape = self.target_shape

        # Downsample amplitude, not complex field, to avoid coherent cancellation.
        amp_full = torch.abs(farfield).to(dtype=torch.float64)
        amp = self._resize_real(amp_full, target_shape)

        if self.use_passed_target and target is not None:
            target_amplitude = self._target_from_argument(target, farfield)
            target_amplitude = self._resize_real(target_amplitude, target_shape)
        else:
            target_amplitude = self._prepare_buffer(self.target_amplitude, farfield)

        weighting_mask = self._prepare_buffer(self.weighting_mask, farfield)

        if amp.shape != target_amplitude.shape:
            raise ValueError(
                f"downsampled farfield shape {tuple(amp.shape)} does not match target shape "
                f"{tuple(target_amplitude.shape)}."
            )

        target_weighted = target_amplitude * weighting_mask
        amp_weighted = amp * weighting_mask

        if self.include_phase:
            phase_full = torch.angle(farfield).to(dtype=torch.float64)
            phase = self._resize_real(phase_full, target_shape)
            phase_reference = self._prepare_buffer(self.phase_reference, farfield)
            numerator = torch.sum(target_weighted * amp_weighted * torch.cos(phase - phase_reference))
        else:
            numerator = torch.sum(target_weighted * amp_weighted)

        norm = torch.sqrt(torch.sum(target_weighted**2) * torch.sum(amp_weighted**2))
        overlap = numerator / torch.clamp(norm, min=np.finfo(float).eps)
        overlap = torch.clamp(overlap, min=0.0, max=1.0)

        return (10**self.cfg.C1) * (1 - overlap) ** 2

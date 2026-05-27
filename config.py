from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Config:
    # Full SLM resolution before superpixel downsampling.
    full_slm_Nx: int = 1920
    full_slm_Ny: int = 1080
    full_cam_Nx: int = 1440
    full_cam_Ny: int = 1080
    superpixel_factor: int = 4
    fourier_padding_factor: int = 2

    # Native SLM pixel size in um.
    base_spix_um: float = 8

    # Native camera pixel size in the optical feedback image plane.
    camera_pixel_size_um: float = 3.45

    # Simulation grid sizes after superpixel grouping.
    Nx: int = 0
    Ny: int = 0

    # Fourier-plane grid.
    NTx: int = 0
    NTy: int = 0

    # Effective SLM pixel size on the simulation grid.
    spix: float = 0.0

    # Input Gaussian 1/e^2 diameters in mm on the SLM plane.
    beam_diameter_x_mm: float = 5.704
    beam_diameter_y_mm: float = 5.938

    # Input beam source: "gaussian", "measured_h5", or "slmsuite_h5".
    input_beam_source: str = "slmsuite_h5"
    input_beam_h5_path: str | None = "/Users/lee/Documents/IQOQI/BeamShaping/FlatHead/10806-SLM-wavefront_superpixel-calibration_00036.h5"
    input_beam_h5_dataset: str = "power"
    measured_power_is_intensity: bool = True
    measured_beam_smoothing_sigma: float = 0.0
    measured_beam_background_percentile: float | None = None

    # Focal length and wavelength used to convert far-field pixels to physical coordinates.
    lens_focal_length_mm: float = 400.0
    wavelength_nm: float = 420.0

    # Target mode: "line_shape" or "rectangle".
    target_mode: str = "rectangle"

    # Shared target center in the Fourier plane. If None, use the array center.
    r0: np.ndarray | None = None

    # Target sizes in um on the focal plane.
    line_width_x_um: float = 100.0
    line_width_y_um: float = 25.0
    rect_width_x_um: float = 500.0
    rect_width_y_um: float = 100.0

    # Gaussian PSF used to blur the ideal target in the focal plane.
    # Use None to estimate from beam size, focal length, SLM sampling, and camera pixels.
    target_psf_sigma_x_um: float | None = None
    target_psf_sigma_y_um: float | None = None
    target_psf_truncate: float = 4.0

    # Optional linear phase tilt in the Fourier plane.
    kx: float = 0.0
    ky: float = 0.0

    # Initial phase guess parameters.
    curv: float = 4.0
    init_phase_d: float = 0.0
    init_phase_asp: float = 0.0
    init_phase_ang: float = np.pi / 4
    init_phase_b: float = 0.0

    # Loss scaling: loss = 10**C1 * (1 - overlap)^2.
    C1: float = 12.0
    C2: float = 0  # efficiency term coefficient

    # Normalize the input beam to a fixed total amplitude scale.
    input_power_norm: float = 10000.0

    # Weighting mask settings.
    weighting_threshold: float = 1e-2
    weighting_background: float = 0.0
    mask_shape: str = "circle"  # "auto", "circle", or "rectangle"
    mask_margin_um: float = 200.0
    output_crop_factor: float = 2.5

    # Optical feedback extraction settings.
    feedback_center_xy_px: tuple[float, float] | None = None
    feedback_crop_shape_yx_px: tuple[int, int] = (300, 300)
    # Counter-clockwise raw camera image rotation before feedback extraction.
    feedback_camera_rotation_deg: int = 0
    feedback_camera_vertical_flip: bool = False
    feedback_camera_horizontal_flip: bool = False
    feedback_angle_guess_deg: float = 0.0
    feedback_angle_search_radius_deg: float = 8.0
    feedback_center_search_radius_px: float = 30.0
    feedback_center_coarse_step_px: float = 2.0
    feedback_scale_search_fraction: float = 0.5
    feedback_edge_threshold_fraction: float = 0.05
    feedback_background_percentile: float = 5.0
    feedback_camera_image_is_intensity: bool = True

    # Optimizer settings.
    optimizer_method: str = "CG"
    optimizer_maxiter: int = 100
    optimizer_disp: bool = True

    # Plotting and saving switches.
    show_initial_summary: bool = True
    show_result_summary: bool = True
    save_root: str = "outputs"
    initial_figure_name: str = "initial_summary.png"
    result_figure_name: str = "result_summary.png"
    bundle_name: str = "run_bundle.pkl"

    # Parameter scan definition.
    scan_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            # "beam_diameter_x_mm": np.linspace(1.5, 3.0, 10),
            # "lens_focal_length_mm": np.linspace(50.0, 300.0, 20),
            "curv": np.linspace(0, 2.0, 20),
            "rect_width_x_um": np.linspace(50.0, 300.0, 20),
            # "rect_width_y_um": np.linspace(5.0, 40.0, 10),
        }
    )
    scan_linked_parameters: dict[str, list[str]] = field(
        default_factory=lambda: {
            "beam_diameter_x_mm": ["beam_diameter_y_mm"],
        }
    )

    def __post_init__(self) -> None:
        self.update_derived()

    def update_derived(self) -> None:
        self.Nx = self.full_slm_Nx // self.superpixel_factor
        self.Ny = self.full_slm_Ny // self.superpixel_factor
        self.spix = self.base_spix_um * self.superpixel_factor
        self.NTx = self.fourier_padding_factor * self.Nx
        self.NTy = self.fourier_padding_factor * self.Ny
        if self.r0 is None:
            self.r0 = np.array([self.NTx / 2, self.NTy / 2], dtype=float)
        else:
            self.r0 = np.array(self.r0, dtype=float)

    def clone(self) -> "Config":
        return Config.from_dict(self.to_dict())

    def apply_updates(self, **updates: Any) -> "Config":
        for key, value in updates.items():
            if key == "sx":
                key = "beam_diameter_x_mm"
                value = 2 * value / 1000
            elif key == "sy":
                key = "beam_diameter_y_mm"
                value = 2 * value / 1000
            setattr(self, key, value)
        self.update_derived()
        return self

    @property
    def sx(self) -> float:
        return self.beam_diameter_x_mm * 1000 / 2

    @sx.setter
    def sx(self, value: float) -> None:
        self.beam_diameter_x_mm = 2 * value / 1000

    @property
    def sy(self) -> float:
        return self.beam_diameter_y_mm * 1000 / 2

    @sy.setter
    def sy(self, value: float) -> None:
        self.beam_diameter_y_mm = 2 * value / 1000

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["r0"] = self.r0.tolist() if self.r0 is not None else None
        data["scan_parameters"] = {
            key: np.asarray(value).tolist() for key, value in self.scan_parameters.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        payload = dict(data)
        if payload.get("r0") is not None:
            payload["r0"] = np.array(payload["r0"], dtype=float)
        payload["scan_parameters"] = {
            key: np.asarray(value, dtype=float) for key, value in payload.get("scan_parameters", {}).items()
        }
        return cls(**payload)


cfg = Config()

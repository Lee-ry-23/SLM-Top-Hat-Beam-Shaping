import os
import pickle

import numpy as np

from config import Config


def create_output_folder(root_dir, run_name):
    output_dir = os.path.join(root_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_save_payload(cfg, run_data):
    optimizer_result = run_data["optimizer_result"]
    return {
        "config": cfg.to_dict(),
        "run_data": {
            "metrics": run_data["metrics"],
            "final_phase": np.asarray(run_data["final_phase"]),
            "fullres_hologram_phase": np.asarray(run_data["fullres_hologram_phase"]),
            "initial_phase": np.asarray(run_data["initial_phase"]),
            "input_beam": np.asarray(run_data["input_beam"]),
            "input_beam_fit": run_data.get("input_beam_fit"),
            "target_amplitude": np.asarray(run_data["target_amplitude"]),
            "reference_phase": np.asarray(run_data["reference_phase"]),
            "weighting_mask": np.asarray(run_data["weighting_mask"]),
            "output_intensity": np.asarray(run_data["output_intensity"]),
            "output_phase": np.asarray(run_data["output_phase"]),
            "loss_history": list(run_data["loss_history"]),
            "iteration_loss_history": list(run_data["iteration_loss_history"]),
            "optimization_time_sec": run_data["optimization_time_sec"],
            "total_time_sec": run_data["total_time_sec"],
            "device": run_data["device"],
            "plot_radius": run_data["plot_radius"],
            "optimizer_result": {
                "success": optimizer_result.success,
                "status": optimizer_result.status,
                "message": optimizer_result.message,
                "nit": getattr(optimizer_result, "nit", None),
                "nfev": getattr(optimizer_result, "nfev", None),
                "fun": getattr(optimizer_result, "fun", None),
            },
        },
    }


def save_run_bundle(output_dir, filename, cfg, run_data):
    payload = build_save_payload(cfg, run_data)
    bundle_path = os.path.join(output_dir, filename)
    with open(bundle_path, "wb") as handle:
        pickle.dump(payload, handle)
    return bundle_path


def load_run_bundle(bundle_path):
    with open(bundle_path, "rb") as handle:
        payload = pickle.load(handle)

    cfg = Config.from_dict(payload["config"])
    run_data = payload["run_data"]
    run_data["config"] = cfg
    return cfg, run_data


def restore_config_and_hologram(bundle_path):
    cfg, run_data = load_run_bundle(bundle_path)
    hologram = np.asarray(run_data.get("fullres_hologram_phase", run_data["final_phase"]))
    return cfg, hologram

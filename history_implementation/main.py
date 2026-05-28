import torch
import numpy as np

from config import cfg
from hologram_compute import cg_optimize
from io_utils import create_output_folder, save_run_bundle
from plotting import create_initial_figure, create_result_figure


def main():
    cfg.update_derived()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    result = cg_optimize(cfg)
    metrics = result["metrics"]

    print(f"Efficiency: {metrics['efficiency'] * 100:.2f}%")
    print("Fidelity:", metrics["fidelity"])
    print(f"RMS Error: {metrics['rms_error'] * 100:.2f}%")
    print(f"Phase Error: {metrics['phase_error'] * 100:.2f}%")
    print(f"Optimization time: {result['optimization_time_sec']:.2f} s")
    print(f"Total time: {result['total_time_sec']:.2f} s")

    if cfg.show_initial_summary:
        create_initial_figure(result, show=True)
    if cfg.show_result_summary:
        create_result_figure(result, show=True)

    save_flag = input("Do you want to save the hologram? (y/n): ").strip().lower()

    if save_flag == 'y':
        run_name = input("Enter output folder name: ").strip()
        output_dir = create_output_folder(cfg.save_root, run_name)

        initial_figure_path = f"{output_dir}/{cfg.initial_figure_name}"
        result_figure_path = f"{output_dir}/{cfg.result_figure_name}"

        create_initial_figure(result, save_path=initial_figure_path, show=False)
        create_result_figure(result, save_path=result_figure_path, show=False)
        bundle_path = save_run_bundle(output_dir, cfg.bundle_name, cfg, result)

        np.save(f"{output_dir}/hologram_phase_fullres.npy", result["fullres_hologram_phase"])
        np.save(f"{output_dir}/hologram_phase_superpixel.npy", np.mod(result["final_phase"], 2 * np.pi))

        print(f"Saved run bundle to {bundle_path}")
        print(f"Saved figures to {output_dir}")
    else:
        print("Hologram not saved.")


if __name__ == "__main__":
    main()

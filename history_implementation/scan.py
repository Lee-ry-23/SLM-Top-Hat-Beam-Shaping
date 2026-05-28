import numpy as np
import itertools
import matplotlib.pyplot as plt

from config import cfg
from hologram_compute import cg_optimize


def expand_scan_points(scan_parameters):
    names = list(scan_parameters.keys())
    value_lists = [list(values) for values in scan_parameters.values()]
    return names, list(itertools.product(*value_lists))


def apply_scan_updates(base_cfg, parameter_names, values):
    run_cfg = base_cfg.clone()
    for name, value in zip(parameter_names, values):
        if name == "sx":
            name = "beam_diameter_x_mm"
            value = 2 * value / 1000
        elif name == "sy":
            name = "beam_diameter_y_mm"
            value = 2 * value / 1000
        setattr(run_cfg, name, value)
        for linked_name in run_cfg.scan_linked_parameters.get(name, []):
            setattr(run_cfg, linked_name, value)
    run_cfg.update_derived()
    return run_cfg


def plot_scan_results(base_cfg, parameter_names, metrics):
    if len(parameter_names) == 1:
        x_values = np.asarray(list(base_cfg.scan_parameters[parameter_names[0]]), dtype=float)

        plt.figure()
        plt.plot(x_values, metrics["efficiency"], marker='o')
        plt.xlabel(parameter_names[0])
        plt.ylabel('Efficiency')
        plt.title('Efficiency Scan')
        plt.grid(True)

        plt.figure()
        plt.plot(x_values, metrics["fidelity"], marker='o')
        plt.xlabel(parameter_names[0])
        plt.ylabel('Fidelity')
        plt.title('Fidelity Scan')
        plt.grid(True)
        return

    if len(parameter_names) == 2:
        x_name, y_name = parameter_names[1], parameter_names[0]
        x_values = np.asarray(list(base_cfg.scan_parameters[x_name]), dtype=float)
        y_values = np.asarray(list(base_cfg.scan_parameters[y_name]), dtype=float)
        shape = (len(y_values), len(x_values))

        plt.figure()
        plt.imshow(
            metrics["efficiency"].reshape(shape),
            origin='lower',
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            aspect='auto',
        )
        plt.colorbar(label='Efficiency')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Efficiency Map')

        plt.figure()
        plt.imshow(
            metrics["fidelity"].reshape(shape),
            origin='lower',
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            aspect='auto',
        )
        plt.colorbar(label='Fidelity')
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Fidelity Map')
        return

    print("Scan has more than 2 dimensions. Metrics were computed but no plot was generated.")


def main():
    parameter_names, scan_points = expand_scan_points(cfg.scan_parameters)
    metrics = {
        "efficiency": np.zeros(len(scan_points)),
        "fidelity": np.zeros(len(scan_points)),
        "rms_error": np.zeros(len(scan_points)),
        "phase_error": np.zeros(len(scan_points)),
    }

    for idx, values in enumerate(scan_points):
        run_cfg = apply_scan_updates(cfg, parameter_names, values)
        label = ", ".join(f"{name}={value:.4g}" for name, value in zip(parameter_names, values))
        print(f"Running {idx + 1}/{len(scan_points)}: {label}")

        result = cg_optimize(run_cfg)
        metrics["efficiency"][idx] = result["efficiency"]
        metrics["fidelity"][idx] = result["fidelity"]
        metrics["rms_error"][idx] = result["rms_error"]
        metrics["phase_error"][idx] = result["phase_error"]

    plot_scan_results(cfg, parameter_names, metrics)
    plt.show()


if __name__ == "__main__":
    main()

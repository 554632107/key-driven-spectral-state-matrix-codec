"""Gramian Angular Field (GAF) from a 1D spectrum."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def min_max_scale_to_minus1_1(x):
    """Scale a 1D signal to [-1, 1]."""
    x = np.asarray(x, dtype=float)
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    return np.clip(2 * (x - x_min) / (x_max - x_min) - 1, -1.0, 1.0)


def resample_1d(x, target_length):
    """Linear resample to a fixed length."""
    x = np.asarray(x, dtype=float)
    if len(x) == target_length:
        return x
    old_idx = np.linspace(0, 1, len(x))
    new_idx = np.linspace(0, 1, target_length)
    return np.interp(new_idx, old_idx, x)


def signal_to_gaf(x, image_size=None, method="summation"):
    """1D spectrum → GAF. method: 'summation' (GASF) or 'difference' (GADF)."""
    x = np.asarray(x, dtype=float)
    if image_size is not None:
        x = resample_1d(x, image_size)

    phi = np.arccos(min_max_scale_to_minus1_1(x))
    if method == "summation":
        return np.cos(phi[:, None] + phi[None, :])
    if method == "difference":
        return np.sin(phi[:, None] - phi[None, :])
    raise ValueError("method must be 'summation' or 'difference'")


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    file_path = os.path.join(_project_root, "dataset", "365_980ex.xlsx")
    label = os.path.splitext(os.path.basename(file_path))[0]
    method = "summation"

    df = pd.read_excel(file_path, header=None)
    spectra = df.iloc[:, 1:].values.T

    gaf_array = np.stack(
        [signal_to_gaf(s, image_size=None, method=method) for s in spectra], axis=0
    )
    print("GAF shape:", gaf_array.shape)

    save_dir = os.path.join(_script_dir, "fig")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(3, gaf_array.shape[0])):
        plt.figure(figsize=(4, 4))
        plt.imshow(gaf_array[i], cmap="Blues", origin="lower", aspect="equal")
        plt.title(f"{label} - Sample {i}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_gaf_{i}.png"), dpi=300, bbox_inches="tight")
        plt.close()

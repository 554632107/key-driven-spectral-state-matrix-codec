"""Markov Transition Field (MTF) from a 1D spectrum."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resample_1d(x, target_length):
    """Linear resample to a fixed length."""
    x = np.asarray(x, dtype=float)
    if len(x) == target_length:
        return x
    old_idx = np.linspace(0, 1, len(x))
    new_idx = np.linspace(0, 1, target_length)
    return np.interp(new_idx, old_idx, x)


def min_max_scale_0_1(x):
    """Per-spectrum min-max to [0, 1]."""
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min)
    return np.zeros_like(x, dtype=float)


def signal_to_mtf(x, n_bins=8, image_size=None):
    """1D spectrum → MTF with n_bins Markov states."""
    x = np.asarray(x, dtype=float)
    if image_size is not None:
        x = resample_1d(x, image_size)

    n = len(x)
    if n < 2:
        return np.zeros((n, n), dtype=float)

    x_norm = min_max_scale_0_1(x)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(x_norm, edges[1:-1], right=False)

    P = np.zeros((n_bins, n_bins), dtype=float)
    for t in range(n - 1):
        P[bin_indices[t], bin_indices[t + 1]] += 1.0

    row_sums = P.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    P[nonzero] = P[nonzero] / row_sums[nonzero]

    return P[bin_indices][:, bin_indices]


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    file_path = os.path.join(_project_root, "dataset", "365_980ex.xlsx")
    label = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path, header=None)
    spectra = df.iloc[:, 1:].values.T

    mtf_array = np.stack(
        [signal_to_mtf(s, n_bins=8, image_size=None) for s in spectra], axis=0
    )
    print("MTF shape:", mtf_array.shape)

    save_dir = os.path.join(_script_dir, "fig")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(3, mtf_array.shape[0])):
        plt.figure(figsize=(4, 4))
        plt.imshow(mtf_array[i], cmap="Greens", origin="lower", aspect="equal")
        plt.title(f"{label} - Sample {i}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_mtf_{i}.png"), dpi=300, bbox_inches="tight")
        plt.close()

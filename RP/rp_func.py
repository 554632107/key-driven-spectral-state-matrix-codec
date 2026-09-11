"""Recurrence Plot (RP) from a 1D spectrum."""

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


def signal_to_rp(x, image_size=None, epsilon=None, binary=False):
    """1D spectrum → RP. If epsilon is None, return the distance matrix."""
    x = np.asarray(x, dtype=float)
    if image_size is not None:
        x = resample_1d(x, image_size)

    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x, dtype=float)

    dist = np.abs(x[:, None] - x[None, :])
    if epsilon is None:
        return dist
    if binary:
        return (dist <= epsilon).astype(float)
    return np.exp(-dist / (epsilon + 1e-8))


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)
    file_path = os.path.join(_project_root, "dataset", "365_980ex.xlsx")
    label = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path, header=None)
    spectra = df.iloc[:, 1:].values.T

    rp_array = np.stack(
        [signal_to_rp(s, image_size=None, epsilon=None, binary=False) for s in spectra],
        axis=0,
    )
    print("RP shape:", rp_array.shape)

    save_dir = os.path.join(_script_dir, "fig")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(3, rp_array.shape[0])):
        plt.figure(figsize=(4, 4))
        plt.imshow(rp_array[i], cmap="Reds", origin="lower", aspect="equal")
        plt.title(f"{label} - Sample {i}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_rp_{i}.png"), dpi=300, bbox_inches="tight")
        plt.close()

"""Build 8-class GAF/RP/MTF images from dataset/*.xlsx → multimodal_dataset_8class.npz."""

import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "GAF"))
sys.path.insert(0, os.path.join(project_root, "RP"))
sys.path.insert(0, os.path.join(project_root, "MTF"))

from gaf_func import signal_to_gaf
from rp_func import signal_to_rp
from mtf_func import signal_to_mtf


def norm01(img):
    """Min-max image to [0, 1]."""
    img = np.asarray(img, dtype=float)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img, dtype=float)


def build_multimodal_dataset_8class():
    """X: (N, 3, 256, 256) = GAF, RP, MTF. y: class 0–7."""
    data_dir = os.path.join(project_root, "dataset")
    target_len = 256
    files_and_labels = [
        ("365ex.xlsx", 0),
        ("980ex.xlsx", 1),
        ("1550ex.xlsx", 2),
        ("365_980ex.xlsx", 3),
        ("365_1550ex.xlsx", 4),
        ("980ex_tm1.xlsx", 5),
        ("365_980ex_tm2.xlsx", 6),
        ("365ex_tm3.xlsx", 7),
    ]

    X_list, y_list = [], []
    for fname, label in files_and_labels:
        file_path = os.path.join(data_dir, fname)
        if not os.path.exists(file_path):
            print(f"skip missing file: {file_path}")
            continue

        print(f"process: {fname} (label={label})")
        df = pd.read_excel(file_path, header=None)
        spectra = df.iloc[:, 1:].values.T

        for spectrum in spectra:
            gaf = signal_to_gaf(spectrum, image_size=target_len, method="summation")
            rp = signal_to_rp(spectrum, image_size=target_len, epsilon=None, binary=False)
            mtf = signal_to_mtf(spectrum, n_bins=8, image_size=target_len)
            X_list.append(np.stack([norm01(gaf), norm01(rp), norm01(mtf)], axis=0))
            y_list.append(label)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\nX={X.shape}, y={y.shape}, counts={np.bincount(y)}")

    save_path = os.path.join(project_root, "multimodal_dataset_8class.npz")
    np.savez_compressed(save_path, X=X, y=y)
    print(f"saved: {save_path}")
    return X, y


if __name__ == "__main__":
    build_multimodal_dataset_8class()

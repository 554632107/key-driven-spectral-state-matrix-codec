"""Plot 8-state matrices: plain codec vs different keys."""

import os
import numpy as np
import matplotlib.pyplot as plt
from spectrum_codec_8_Secure import SpectrumCodec8, SpectrumCodec8Secure

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

STATE_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e91e63",
    "#795548",
]


def _draw_grid(ax, matrix_size):
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(matrix_size))
    ax.set_yticks(np.arange(matrix_size))
    ax.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)


def visualize_encoded_matrix_8(message: str, matrix_size: int = 10):
    """Plain (no-key) encoding: matrix + state histogram."""
    codec = SpectrumCodec8()
    encoded, original_bits = codec.encode_message(message, matrix_size=matrix_size)
    cmap = plt.matplotlib.colors.ListedColormap(STATE_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im1 = axes[0].imshow(encoded, cmap=cmap, vmin=0, vmax=7)
    axes[0].set_title(f"8-state matrix ({matrix_size}x{matrix_size})", fontweight="bold", fontsize=14)
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")
    _draw_grid(axes[0], matrix_size)
    cbar = plt.colorbar(im1, ax=axes[0], ticks=range(8))
    cbar.ax.set_yticklabels(codec.state_names, fontsize=9)

    unique, counts = np.unique(encoded, return_counts=True)
    bars = axes[1].bar(
        [codec.state_names[i] for i in unique],
        counts,
        color=[STATE_COLORS[i] for i in unique],
        alpha=0.8,
        edgecolor="black",
    )
    axes[1].set_title("state counts", fontweight="bold", fontsize=14)
    axes[1].tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{count}", ha="center", va="bottom")

    axes[1].text(
        0.98,
        0.97,
        f"{original_bits} bit = {original_bits // 8} bytes\n{len(unique)}/8 states\n3 bit/cell",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )
    fig.suptitle(f'message: "{message}" ({len(message)} chars)', fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum_encoding")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "encoding_visualization_8.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.close(fig)


def compare_secure_encodings_8(message: str, matrix_size: int = 10, keys=None):
    """Same message under different keys."""
    if not keys:
        keys = ["key_A", "key_B"]

    cmap = plt.matplotlib.colors.ListedColormap(STATE_COLORS)
    fig = plt.figure(figsize=(5 * len(keys) + 1.2, 5))
    axes = [fig.add_subplot(1, len(keys), i + 1) for i in range(len(keys))]

    im = None
    for ax, key in zip(axes, keys):
        encoded, _ = SpectrumCodec8Secure(key=key).encode_message(message, matrix_size=matrix_size)
        im = ax.imshow(encoded, cmap=cmap, vmin=0, vmax=7)
        ax.set_title(f'key="{key}"', fontweight="bold", fontsize=12)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        _draw_grid(ax, matrix_size)

    plt.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=range(8))
    cbar.ax.set_yticklabels(SpectrumCodec8().state_names, fontsize=9)
    fig.suptitle(f'same message, different keys\n"{message}"', fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.90, 0.93])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum_encoding")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "encoding_secure_compare_8.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    compare_secure_encodings_8("吉林大学", matrix_size=10, keys=["key_A", "key_B", "another_key"])

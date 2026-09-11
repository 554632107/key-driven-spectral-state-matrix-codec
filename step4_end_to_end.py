"""End-to-end test: encode with a key → CNN recognize → decode with the same key."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_codec_8_Secure import SpectrumCodec8Secure

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FONT_SIZES = {
    "title": 18,
    "axis_label": 16,
    "tick": 16,
    "colorbar": 14,
    "annotation": 14,
    "suptitle": 20,
}

project_root = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(project_root, "multimodal_dataset_8class.npz")
MODEL_PATH = os.path.join(project_root, "multimodal_cnn_8class.pt")
STATE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e91e63", "#795548"]


class SimpleCNN(nn.Module):
    """Same architecture as step2."""

    def __init__(self, num_classes=8, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_multimodal_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"dataset not found: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    X, y = data["X"], data["y"]
    print(f"dataset X={X.shape} y={y.shape}")
    indices_by_label = {label: np.where(y == label)[0] for label in range(8)}
    return X, y, indices_by_label


def load_cnn_model(device="cpu"):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    model = SimpleCNN(num_classes=8, in_channels=3)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"model: {MODEL_PATH}")
    return model


def create_multimodal_images(encoded_matrix, X, indices_by_label):
    """Pick one real sample image per cell from that cell's class."""
    H, W = encoded_matrix.shape
    _, C, H_img, W_img = X.shape
    images = np.zeros((H * W, C, H_img, W_img), dtype=np.float32)
    for idx, label in enumerate(encoded_matrix.flatten()):
        images[idx] = X[np.random.choice(indices_by_label[int(label)])]
    return images


def add_grid(ax, matrix_size):
    if matrix_size <= 10:
        step = 1
    elif matrix_size <= 20:
        step = 2
    elif matrix_size <= 50:
        step = 5
    else:
        step = 10
    ticks = np.arange(0, matrix_size, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)


def visualize_encoding_process(message, key, matrix_size, codec, save_path):
    """Logical matrix vs position-scrambled matrix."""
    encoded_matrix, _ = codec.encode_message(message, matrix_size=matrix_size)

    binary = "".join(format(b, "08b") for b in message.encode("utf-8"))
    binary += "0" * ((3 - len(binary) % 3) % 3)
    logical_states = [codec.encode_map.get(binary[i : i + 3], 0) for i in range(0, len(binary), 3)]
    while len(logical_states) < matrix_size * matrix_size:
        logical_states.append(0)
    logical_matrix = np.array(logical_states).reshape(matrix_size, matrix_size)

    cmap = plt.matplotlib.colors.ListedColormap(STATE_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(logical_matrix, cmap=cmap, vmin=0, vmax=7)
    axes[0].set_title("Step 1: Logical Matrix\n(Before Scrambling)", fontsize=FONT_SIZES["title"], fontweight="bold")
    axes[0].set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    axes[0].set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    axes[0].tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(axes[0], matrix_size)

    axes[1].axis("off")
    axes[1].text(
        0.5, 0.7, "Position\nScrambling", ha="center", va="center",
        fontsize=FONT_SIZES["annotation"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )
    axes[1].annotate("", xy=(0.85, 0.45), xytext=(0.15, 0.45), arrowprops=dict(arrowstyle="->", lw=2.5, color="black"))
    axes[1].text(0.5, 0.2, f"Key: '{key}'", ha="center", va="center", fontsize=FONT_SIZES["annotation"], style="italic")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    im3 = axes[2].imshow(encoded_matrix, cmap=cmap, vmin=0, vmax=7)
    axes[2].set_title("Step 2: Encrypted Matrix\n(After Scrambling)", fontsize=FONT_SIZES["title"], fontweight="bold")
    axes[2].set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    axes[2].set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    axes[2].tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(axes[2], matrix_size)

    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im3, cax=cbar_ax, ticks=range(8))
    cbar.ax.set_yticklabels(codec.state_names, fontsize=FONT_SIZES["colorbar"])
    fig.suptitle("Encoding Process Visualization", fontsize=FONT_SIZES["suptitle"], fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"   saved: {save_path}")
    plt.close(fig)


def visualize_prediction_comparison(original_matrix, predicted_matrix, codec, accuracy, save_path):
    """Original vs CNN prediction vs errors."""
    cmap = plt.matplotlib.colors.ListedColormap(STATE_COLORS)
    error_matrix = (original_matrix != predicted_matrix).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(original_matrix, cmap=cmap, vmin=0, vmax=7)
    axes[0].set_title("Original Encrypted Matrix\n(Sender)", fontsize=FONT_SIZES["title"], fontweight="bold")
    axes[0].set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    axes[0].set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    axes[0].tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(axes[0], original_matrix.shape[0])

    im2 = axes[1].imshow(predicted_matrix, cmap=cmap, vmin=0, vmax=7)
    axes[1].set_title(f"CNN Predicted Matrix\n(Accuracy: {accuracy:.2f}%)", fontsize=FONT_SIZES["title"], fontweight="bold")
    axes[1].set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    axes[1].set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    axes[1].tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(axes[1], predicted_matrix.shape[0])

    axes[2].imshow(predicted_matrix, cmap=cmap, vmin=0, vmax=7, alpha=0.6)
    error_rows, error_cols = np.where(error_matrix == 1)
    if len(error_rows) > 0:
        axes[2].scatter(error_cols, error_rows, s=200, c="red", marker="x", linewidths=3)
        for row, col in zip(error_rows, error_cols):
            axes[2].text(col, row, "X", color="red", fontsize=FONT_SIZES["annotation"], ha="center", va="center", fontweight="bold")
    axes[2].set_title(f"Error Marking\n({len(error_rows)}/{error_matrix.size})", fontsize=FONT_SIZES["title"], fontweight="bold")
    axes[2].set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    axes[2].set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    axes[2].tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(axes[2], original_matrix.shape[0])

    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, ticks=range(8))
    cbar.ax.set_yticklabels(codec.state_names, fontsize=FONT_SIZES["colorbar"])
    fig.suptitle("CNN Recognition Result Comparison", fontsize=FONT_SIZES["suptitle"], fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"   saved: {save_path}")
    plt.close(fig)


def _wrap_three_lines(text):
    words = text.split()
    if not words:
        return text
    a, b = len(words) // 3, 2 * len(words) // 3
    return " ".join(words[:a]) + "\n" + " ".join(words[a:b]) + "\n" + " ".join(words[b:])


def visualize_decoding_result(original_message, decoded_message, original_matrix, predicted_matrix, key, save_path):
    """Text match plus the two matrices."""
    cmap = plt.matplotlib.colors.ListedColormap(STATE_COLORS)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 3], hspace=0.3, wspace=0.3)
    match = original_message == decoded_message
    status = "Success" if match else "Failed"

    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")
    ax1.text(
        0.5, 0.4, f"Sender Input (Original Message):\n'{_wrap_three_lines(original_message)}'",
        ha="center", va="center", fontsize=FONT_SIZES["annotation"],
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen", alpha=0.8), wrap=True,
    )
    ax1.text(0.08, 0.95, "Sender", fontsize=FONT_SIZES["title"], fontweight="bold", color="blue")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.axis("off")
    ax2.text(
        0.5, 0.4, f"Receiver Output (Decoded Message):\n'{_wrap_three_lines(decoded_message)}'",
        ha="center", va="center", fontsize=FONT_SIZES["annotation"],
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen" if match else "lightcoral", alpha=0.8), wrap=True,
    )
    ax2.text(
        0.08, 0.95, f"Receiver ({status})", fontsize=FONT_SIZES["title"],
        fontweight="bold", color="green" if match else "red",
    )

    fig.text(0.48, 0.70, "↓", ha="center", fontsize=24, color="gray")
    fig.text(0.52, 0.70, f"Key: '{key}'", ha="left", fontsize=FONT_SIZES["annotation"], style="italic", color="#FF4500")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.imshow(original_matrix, cmap=cmap, vmin=0, vmax=7)
    ax3.set_title("Original Encrypted Matrix", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax3.set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    ax3.set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    ax3.tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(ax3, original_matrix.shape[0])

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.imshow(predicted_matrix, cmap=cmap, vmin=0, vmax=7)
    accuracy = (original_matrix == predicted_matrix).sum() / original_matrix.size * 100
    ax4.set_title(f"CNN Predicted Matrix (Accuracy: {accuracy:.1f}%)", fontsize=FONT_SIZES["title"], fontweight="bold")
    ax4.set_xlabel("Column Index", fontsize=FONT_SIZES["axis_label"])
    ax4.set_ylabel("Row Index", fontsize=FONT_SIZES["axis_label"])
    ax4.tick_params(labelsize=FONT_SIZES["tick"])
    add_grid(ax4, predicted_matrix.shape[0])

    fig.suptitle("End-to-End Decoding Result", fontsize=FONT_SIZES["suptitle"], fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"   saved: {save_path}")
    plt.close(fig)


def simulate_sender(message, key, matrix_size=10):
    print("\n" + "=" * 70)
    print("sender")
    print("=" * 70)
    print(f"message: '{message}'")
    print(f"key: '{key}'")
    codec = SpectrumCodec8Secure(key=key)
    encoded_matrix, original_bits = codec.encode_message(message, matrix_size=matrix_size)
    print(f"matrix {encoded_matrix.shape}, {original_bits} bits")
    return codec, encoded_matrix, original_bits


def simulate_transmission(encoded_matrix, X, indices_by_label, model, device):
    print("\n" + "=" * 70)
    print("recognize")
    print("=" * 70)
    images = create_multimodal_images(encoded_matrix, X, indices_by_label)
    with torch.no_grad():
        predictions = model(torch.from_numpy(images).to(device)).argmax(dim=1).cpu().numpy()
    predicted_matrix = predictions.reshape(encoded_matrix.shape)
    correct = (predicted_matrix == encoded_matrix).sum()
    accuracy = correct / encoded_matrix.size * 100
    print(f"CNN acc: {correct}/{encoded_matrix.size} = {accuracy:.2f}%")
    return predicted_matrix, accuracy


def simulate_receiver(predicted_matrix, original_bits, key):
    print("\n" + "=" * 70)
    print("receiver")
    print("=" * 70)
    decoded_message = SpectrumCodec8Secure(key=key).decode_matrix(predicted_matrix, original_bits=original_bits)
    print(f"decoded: '{decoded_message}'")
    return decoded_message


def test_wrong_key_attack(predicted_matrix, original_bits, correct_key, wrong_keys):
    print("\n" + "=" * 70)
    print(f"wrong-key test  correct='{correct_key}'")
    print("=" * 70)
    for wrong_key in wrong_keys:
        print(f"\ntry '{wrong_key}'")
        try:
            decoded_wrong = SpectrumCodec8Secure(key=wrong_key).decode_matrix(
                predicted_matrix, original_bits=original_bits
            )
            print(f"result: '{decoded_wrong}'")
        except Exception as e:
            print(f"failed: {e}")


def run_end_to_end_secure_test():
    print("\n" + "=" * 70)
    print("8-state secure end-to-end test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    X, _, indices_by_label = load_multimodal_dataset()
    model = load_cnn_model(device)

    vis_dir = os.path.join(project_root, "spectrum_encoding", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    message = (
        "This research presents a triband metasurface that independently manipulates light "
        "across visible, midwave infrared (MWIR), and long-wave infrared (LWIR) wavelengths, "
        "enhancing optical security through advanced encryption techniques."
    )
    key = "JLU_2026_Secret"
    matrix_size = 30

    codec, encoded_matrix, original_bits = simulate_sender(message, key, matrix_size)
    visualize_encoding_process(
        message, key, matrix_size, codec, os.path.join(vis_dir, "case1_encoding_process.png")
    )
    predicted_matrix, accuracy = simulate_transmission(encoded_matrix, X, indices_by_label, model, device)
    visualize_prediction_comparison(
        encoded_matrix, predicted_matrix, codec, accuracy, os.path.join(vis_dir, "case1_prediction_comparison.png")
    )
    decoded_message = simulate_receiver(predicted_matrix, original_bits, key)
    visualize_decoding_result(
        message, decoded_message, encoded_matrix, predicted_matrix, key,
        os.path.join(vis_dir, "case1_decoding_result.png"),
    )

    print("\n" + "-" * 70)
    if message == decoded_message:
        print("match")
    else:
        errors = sum(1 for a, b in zip(message, decoded_message) if a != b)
        errors += abs(len(message) - len(decoded_message))
        cer = errors / max(len(message), len(decoded_message)) * 100
        print(f"mismatch: {errors} chars, CER {cer:.2f}%")

    test_wrong_key_attack(predicted_matrix, original_bits, key, ["wrong_key_1", "hacker_key", "key_B"])
    print(f"\nfigures: {vis_dir}")
    print("same key required; wrong key yields garbage; CNN errors affect decode quality")


if __name__ == "__main__":
    run_end_to_end_secure_test()

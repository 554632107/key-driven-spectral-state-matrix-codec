"""Train an 8-class CNN on GAF/RP/MTF images → multimodal_cnn_8class.pt."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

project_root = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DATASETS = [
    "multimodal_dataset_8class.npz",
    "multimodal_dataset.npz",
]


class MultimodalDataset(Dataset):
    """Subset of X[N,C,H,W], y[N] by index list."""

    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = torch.from_numpy(self.X[real_idx].astype(np.float32))
        y = torch.tensor(int(self.y[real_idx]), dtype=torch.long)
        return x, y


class SimpleCNN(nn.Module):
    """Small CNN: 4 conv blocks + global pool."""

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


def find_dataset_path():
    for name in CANDIDATE_DATASETS:
        path = os.path.join(project_root, name)
        if os.path.exists(path):
            print(f"dataset: {path}")
            return path
    raise FileNotFoundError(f"none of {CANDIDATE_DATASETS} found")


def load_dataset():
    data = np.load(find_dataset_path())
    X, y = data["X"], data["y"]
    print("X shape:", X.shape, "y shape:", y.shape)
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c}")
    return X, y


def split_indices_3way(n_samples, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Shuffle split 6:2:2 (not stratified)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total


def get_all_predictions(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            outputs = model(x.to(device))
            all_preds.append(outputs.argmax(1).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    batch_size, num_epochs, lr = 16, 40, 1e-3
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    X, y = load_dataset()
    n_samples, C, _, _ = X.shape
    num_classes = len(np.unique(y))
    print(f"num_classes: {num_classes}")
    if num_classes != 8:
        print("warning: label count is not 8")

    train_idx, val_idx, test_idx = split_indices_3way(n_samples, 0.6, 0.2, seed=123)
    print(f"train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    train_loader = DataLoader(MultimodalDataset(X, y, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(MultimodalDataset(X, y, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(MultimodalDataset(X, y, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=num_classes, in_channels=C).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_model_state = None
    print("\ntrain 8-class CNN")
    print("=" * 70)

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:3d}/{num_epochs}] "
                f"train {train_loss:.4f}/{train_acc:.4f} | "
                f"val {val_loss:.4f}/{val_acc:.4f} | best {best_val_acc:.4f}"
            )

    print("=" * 70)
    print(f"best val acc: {best_val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\ntest loss {test_loss:.4f}  acc {test_acc:.4f}")
    test_preds, test_labels = get_all_predictions(model, test_loader, device)
    print(confusion_matrix(test_labels, test_preds))
    print(classification_report(test_labels, test_preds, digits=4))

    save_path = os.path.join(project_root, "multimodal_cnn_8class.pt")
    torch.save(model.state_dict(), save_path)
    print("saved:", save_path)


if __name__ == "__main__":
    main()

"""
Train VOC2007 multi-label model (backbone + head). Saves checkpoints and history for visualization.
"""
import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam

from config import (
    CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    NUM_CLASSES
)
from dataset import get_train_val_loaders
from model import build_model


def binary_accuracy(pred, target, thresh=0.5):
    """Multi-label accuracy: fraction of correct labels (threshold 0.5)."""
    pred_bin = (pred >= thresh).float()
    return (pred_bin == target).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    non_blocking = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += binary_accuracy(out, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    non_blocking = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_acc += binary_accuracy(out, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except (AssertionError, RuntimeError):
            device = torch.device("cpu")
            print("CUDA not available (e.g. CPU-only PyTorch), using CPU")
    print(f"Device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        batch_size=BATCH_SIZE, num_workers=2, val_ratio=0.1, use_cuda=(device.type == "cuda")
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model().to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "voc2007_best.pt")
    last_path = os.path.join(CHECKPOINT_DIR, "voc2007_last.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "val_acc": val_acc, "history": history},
                best_path,
            )
            print(f"  -> Saved best to {best_path}")

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "history": history},
            last_path,
        )

    history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    main()

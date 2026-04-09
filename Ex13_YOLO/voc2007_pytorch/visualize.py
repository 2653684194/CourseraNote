"""
Visualize VOC2007 training: loss/accuracy curves and sample predictions.
"""
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR, VIS_DIR, VOC_CLASSES, NUM_CLASSES, IMAGE_SIZE
from dataset import get_train_val_loaders, VOC2007Dataset, build_image_list
from model import build_model


def plot_curves(history_path=None, save_dir=VIS_DIR):
    """Plot train/val loss and accuracy from history.json or from last checkpoint."""
    if history_path is None:
        history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    if not os.path.isfile(history_path):
        ckpt = os.path.join(CHECKPOINT_DIR, "voc2007_last.pt")
        if os.path.isfile(ckpt):
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            history = data.get("history", {})
        else:
            print("No history.json or checkpoint found. Run train.py first.")
            return
    else:
        with open(history_path, "r") as f:
            history = json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Multi-label accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "train_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curves saved to {out_path}")


def visualize_predictions(num_samples=8, thresh=0.5, save_dir=VIS_DIR):
    """Load best model, run on val set, plot images with predicted vs true labels."""
    from torchvision import transforms

    ckpt_path = os.path.join(CHECKPOINT_DIR, "voc2007_best.pt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "voc2007_last.pt")
    if not os.path.isfile(ckpt_path):
        print("No checkpoint found. Run train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = get_train_val_loaders(batch_size=num_samples, num_workers=0, val_ratio=0.1)
    x, y = next(iter(val_loader))
    x_dev = x.to(device)
    with torch.no_grad():
        pred = model(x_dev).cpu().numpy()
    y_np = y.numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def denorm(t):
        t = t.permute(1, 2, 0).numpy()
        t = t * std + mean
        return np.clip(t, 0, 1)

    nrow = 2
    ncol = (num_samples + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    if nrow * ncol == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        img = denorm(x[i])
        ax.imshow(img)
        true_labels = [VOC_CLASSES[j] for j in range(NUM_CLASSES) if y_np[i, j] > 0.5]
        pred_labels = [VOC_CLASSES[j] for j in range(NUM_CLASSES) if pred[i, j] >= thresh]
        ax.set_title(f"True: {true_labels}\nPred: {pred_labels}", fontsize=8)
        ax.axis("off")
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    plot_curves()
    visualize_predictions(num_samples=8)

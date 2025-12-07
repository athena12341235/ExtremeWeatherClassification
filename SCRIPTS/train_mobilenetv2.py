#!/usr/bin/env python3
"""
Fine-tune a pretrained MobileNetV2 for Extreme vs Normal weather classification
and/Users/aryanthodupunuri/extreme-weather-classification-1/.venv/bin/python SCRIPTS/train_mobilenetv2.py \
  --split-root dataset_split \
  --epochs 10 \
  --batch-size 64 \
  --out-dir MODELS \
  --num-workers 2 produce Grad-CAM visualizations.

Features:
- Uses torchvision MobileNetV2 pretrained on ImageNet; replaces classifier for 2 classes.
- Optionally freezes feature extractor for initial epochs.
- ImageNet normalization, data augmentation via RandomResizedCrop/Flip.
- Saves best checkpoint, metrics CSV, confusion matrix, and classification report.
- Grad-CAM heatmaps for a sample of validation images to highlight influential regions.

Usage examples:
  # From single folder (random splits)
  python SCRIPTS/train_mobilenetv2.py \
      --data-root DATA/raw_data/cleaned_data \
      --epochs 10 --batch-size 64 --out-dir MODELS

  # From existing split
  python SCRIPTS/train_mobilenetv2.py \
      --split-root dataset_split \
      --epochs 10 --batch-size 64 --out-dir MODELS

  # Enable Grad-CAM generation (after training best checkpoint)
  python SCRIPTS/train_mobilenetv2.py --split-root dataset_split --gradcam_mobilenetv2 --gradcam_mobilenetv2-samples 12

  # Dry-run to sanity-check a forward/backward step
  python SCRIPTS/train_mobilenetv2.py --dry-run --limit-per-class 8 --num-workers 0
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore

import matplotlib.pyplot as plt
import csv

CLASSES = ("extreme", "normal")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # De-normalizer for visualization
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1.0/s for s in std]
    )
    return train_tf, val_tf


def make_datasets(
    data_root: Path,
    split_root: Optional[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    limit_per_class: Optional[int] = None,
    img_size: int = 224,
):
    train_tf, val_tf = build_transforms(img_size)

    if split_root and (split_root / "train").exists() and (split_root / "val").exists():
        train_ds = torchvision.datasets.ImageFolder(split_root / "train", transform=train_tf)
        val_ds = torchvision.datasets.ImageFolder(split_root / "val", transform=val_tf)
        test_ds = torchvision.datasets.ImageFolder(split_root / "test", transform=val_tf) if (split_root / "test").exists() else val_ds
        return train_ds, val_ds, test_ds

    # Fallback: single folder -> split
    full_ds = torchvision.datasets.ImageFolder(data_root, transform=train_tf)

    if limit_per_class is not None and limit_per_class > 0:
        indices_by_class: Dict[int, List[int]] = {i: [] for i in range(len(full_ds.classes))}
        for idx, (_, label) in enumerate(full_ds.samples):
            if len(indices_by_class[label]) < limit_per_class:
                indices_by_class[label].append(idx)
        limited_indices: List[int] = []
        for lbl in sorted(indices_by_class.keys()):
            limited_indices.extend(indices_by_class[lbl])
        full_ds = Subset(full_ds, limited_indices)  # type: ignore

    n = len(full_ds)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    def set_subset_transform(sub: Subset, tf):  # type: ignore
        if isinstance(sub.dataset, torchvision.datasets.ImageFolder):
            sub.dataset.transform = tf
        else:
            sub.dataset.dataset.transform = tf  # type: ignore[attr-defined]

    set_subset_transform(train_ds, train_tf)
    set_subset_transform(val_ds, val_tf)
    set_subset_transform(test_ds, val_tf)

    return train_ds, val_ds, test_ds


def create_mobilenet_v2(num_classes: int = 2, pretrained: bool = True, freeze_features: bool = False):
    # Use new weights API if available
    try:
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.mobilenet_v2(weights=weights)
    except Exception:
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
    return model


def train_one_epoch(model, loader, device, optimizer, criterion) -> Tuple[float, float]:
    model.train()
    running, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    running, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return running / max(total, 1), correct / max(total, 1), all_preds, all_labels


def save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: Path):
    if confusion_matrix is None:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------- Grad-CAM ----------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def __call__(self, input_tensor: torch.Tensor, target_index: Optional[int] = None) -> np.ndarray:
        """
        Returns CAM heatmap as HxW numpy array scaled to [0,1].
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)  # shape [1, C]
        if target_index is None:
            target_index = int(logits.argmax().item())
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_index] = 1.0
        logits.backward(gradient=one_hot)

        assert self.activations is not None and self.gradients is not None, "Hooks did not capture activations/gradients"
        grads = self.gradients  # [B, K, H, W]
        acts = self.activations  # [B, K, H, W]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()


def find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last_conv


def overlay_heatmap_on_image(img_tensor: torch.Tensor, heatmap: np.ndarray) -> Image.Image:
    # img_tensor is normalized; de-normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    base = Image.fromarray(img_np)

    # Resize heatmap to image size
    hmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(base.size, resample=Image.BILINEAR)
    hmap_np = np.array(hmap, dtype=np.float32) / 255.0
    # Apply colormap (simple jet-like)
    cmap = plt.get_cmap('jet')
    colored = (cmap(hmap_np)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(colored)

    # Overlay with alpha
    overlay = Image.blend(base.convert('RGBA'), heat_img.convert('RGBA'), alpha=0.45)
    return overlay.convert('RGB')


def generate_gradcam_samples(model: nn.Module, val_loader: DataLoader, device, out_dir: Path, num_samples: int = 8):
    ensure_dir(out_dir)
    target_layer = find_last_conv(model.features)
    cam = GradCAM(model, target_layer)
    saved = 0
    model.eval()
    with torch.enable_grad():
        for images, labels in val_loader:
            for i in range(images.size(0)):
                if saved >= num_samples:
                    cam.remove_hooks()
                    return
                img = images[i:i+1].to(device)
                logits = model(img)
                pred = int(logits.argmax(1).item())
                heatmap = cam(img, target_index=pred)
                overlay = overlay_heatmap_on_image(images[i], heatmap)
                fname = out_dir / f"gradcam_{saved:03d}_pred-{CLASSES[pred]}_true-{CLASSES[labels[i].item()]}.png"
                overlay.save(fname)
                saved += 1
    cam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 with optional Grad-CAM visualization")
    parser.add_argument("--data-root", type=str, default=str(Path("DATA")/"raw_data"/"cleaned_data"))
    parser.add_argument("--split-root", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="MODELS")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--freeze-features", action="store_true", help="Freeze feature extractor parameters")
    parser.add_argument("--gradcam_mobilenetv2", action="store_true", help="Generate Grad-CAM images on val set after training")
    parser.add_argument("--gradcam_mobilenetv2-samples", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = Path(args.split_root) if args.split_root else None
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    report_dir = Path("reports")/"training"; ensure_dir(report_dir)
    gradcam_dir = report_dir/"gradcam_mobilenetv2"; ensure_dir(gradcam_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = args.limit_per_class if args.limit_per_class and args.limit_per_class > 0 else None
    train_ds, val_ds, test_ds = make_datasets(
        data_root=data_root,
        split_root=split_root,
        val_ratio=0.15,
        test_ratio=0.15,
        limit_per_class=limit,
        img_size=args.img_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = create_mobilenet_v2(num_classes=2, pretrained=True, freeze_features=args.freeze_features).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only train classifier if features frozen; else whole model
    params = model.classifier.parameters() if args.freeze_features else model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.dry_run:
        model.train()
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("MobileNetV2 dry-run ok. Loss=", float(loss.item()))
        return

    best_val_acc = 0.0
    best_model_path = out_dir / "mobilenetv2_best.pth"
    metrics_csv = out_dir / "mobilenetv2_metrics.csv"

    with metrics_csv.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])
            print(f"Epoch {epoch:02d}/{args.epochs} | train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch,
                    'val_acc': best_val_acc,
                    'config': vars(args),
                }, best_model_path)

    # Load best and evaluate on test
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, device, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")
    save_confusion_matrix(y_true, y_pred, class_names=list(CLASSES), out_path=report_dir/"mobilenetv2_confusion_matrix.png")
    report_txt = report_dir / "mobilenetv2_classification_report.txt"
    if classification_report is not None:
        rep = classification_report(y_true, y_pred, target_names=list(CLASSES))
        report_txt.write_text(rep)

    if args.gradcam:
        print("Generating Grad-CAM visualizations...")
        generate_gradcam_samples(model, val_loader, device, out_dir=gradcam_dir, num_samples=args.gradcam_samples)
        print(f"Grad-CAM saved to: {gradcam_dir}")

    print(f"Training complete. Best model: {best_model_path} | Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()

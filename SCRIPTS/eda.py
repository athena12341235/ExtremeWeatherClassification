#!/usr/bin/env python3
"""
EDA for cleaned Extreme vs Normal weather dataset.
Reads from cleaned_data and dataset_split and writes figures and CSV summaries to OUTPUT/EDA.

Outputs (under OUTPUT/EDA):
- counts_cleaned.csv: per-class counts in cleaned_data
- counts_split.csv: per-split, per-class counts in dataset_split
- filetypes.csv: per-class file extension distribution in cleaned_data
- channel_stats.csv: per-class RGB mean/std and brightness mean/std (on a sample)
- brightness_hist_extreme.png / brightness_hist_normal.png: brightness histograms
- mean_image_extreme.png / mean_image_normal.png: per-class mean image (on a sample)
- montage_extreme.png / montage_normal.png: sample grids

Usage:
  python SCRIPTS/eda.py [--clean-root cleaned_data] [--split-root dataset_split] [--sample-per-class N]
"""

from __future__ import annotations
import os
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ("extreme", "normal")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTS


def read_image_rgb(path: Path) -> np.ndarray | None:
    try:
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im)
                return arr
    except Exception:
        return None


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def list_cleaned_images(clean_root: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {c: [] for c in CLASSES}
    for c in CLASSES:
        class_dir = clean_root / c
        if class_dir.exists():
            out[c] = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
    return out


def counts_cleaned(clean_root: Path) -> Dict[str, int]:
    imgs = list_cleaned_images(clean_root)
    return {c: len(imgs.get(c, [])) for c in CLASSES}


def filetype_distribution(clean_root: Path) -> Dict[str, Dict[str, int]]:
    dist: Dict[str, Dict[str, int]] = {c: {} for c in CLASSES}
    imgs = list_cleaned_images(clean_root)
    for c, paths in imgs.items():
        for p in paths:
            ext = p.suffix.lower()
            dist[c][ext] = dist[c].get(ext, 0) + 1
    return dist


def counts_split(split_root: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        out[split] = {}
        for c in CLASSES:
            d = split_root / split / c
            n = 0
            if d.exists():
                n = len([p for p in d.iterdir() if p.is_file() and is_image(p)])
            out[split][c] = n
    return out


def sample_paths(paths: List[Path], k: int) -> List[Path]:
    if k <= 0 or k >= len(paths):
        return list(paths)
    return random.sample(paths, k)


def compute_channel_stats(clean_root: Path, sample_per_class: int) -> Dict[str, Dict[str, float]]:
    imgs = list_cleaned_images(clean_root)
    stats: Dict[str, Dict[str, float]] = {}
    for c in CLASSES:
        paths = sample_paths(imgs.get(c, []), sample_per_class)
        if not paths:
            continue
        count = 0
        sum_rgb = np.zeros(3, dtype=np.float64)
        sumsq_rgb = np.zeros(3, dtype=np.float64)
        sum_brightness = 0.0
        sumsq_brightness = 0.0
        for p in paths:
            arr = read_image_rgb(p)
            if arr is None:
                continue
            # Images should be 224x224x3 uint8
            arrf = arr.astype(np.float64) / 255.0
            # per-image stats
            mean_rgb = arrf.reshape(-1, 3).mean(axis=0)
            brightness = arrf.mean()  # simple avg over all pixels & channels
            sum_rgb += mean_rgb
            sumsq_rgb += mean_rgb ** 2
            sum_brightness += brightness
            sumsq_brightness += brightness ** 2
            count += 1
        if count == 0:
            continue
        mean_rgb = (sum_rgb / count).tolist()
        var_rgb = (sumsq_rgb / count - (sum_rgb / count) ** 2).tolist()
        std_rgb = [float(np.sqrt(max(v, 0.0))) for v in var_rgb]
        mean_b = float(sum_brightness / count)
        var_b = float(sumsq_brightness / count - (sum_brightness / count) ** 2)
        std_b = float(np.sqrt(max(var_b, 0.0)))
        stats[c] = {
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
            "std_r": float(std_rgb[0]),
            "std_g": float(std_rgb[1]),
            "std_b": float(std_rgb[2]),
            "brightness_mean": mean_b,
            "brightness_std": std_b,
            "n_images": count,
        }
    return stats


def plot_brightness_hist(clean_root: Path, report_dir: Path, sample_per_class: int = 500, bins: int = 40):
    imgs = list_cleaned_images(clean_root)
    for c in CLASSES:
        paths = sample_paths(imgs.get(c, []), sample_per_class)
        vals: List[float] = []
        for p in paths:
            arr = read_image_rgb(p)
            if arr is None:
                continue
            vals.append(float(arr.mean() / 255.0))
        if vals:
            plt.figure(figsize=(6,4))
            plt.hist(vals, bins=bins, color=(0.2,0.4,0.8) if c=="extreme" else (0.8,0.4,0.2), alpha=0.8)
            plt.title(f"Brightness histogram ({c})")
            plt.xlabel("Brightness (0-1)")
            plt.ylabel("Count")
            plt.tight_layout()
            out = report_dir / f"brightness_hist_{c}.png"
            plt.savefig(out, dpi=150)
            plt.close()


def save_mean_image(clean_root: Path, report_dir: Path, c: str, sample_per_class: int = 500):
    paths = list_cleaned_images(clean_root).get(c, [])
    paths = sample_paths(paths, sample_per_class)
    if not paths:
        return
    acc = None
    n = 0
    for p in paths:
        arr = read_image_rgb(p)
        if arr is None:
            continue
        arrf = arr.astype(np.float32) / 255.0
        if acc is None:
            acc = np.zeros_like(arrf, dtype=np.float64)
        acc += arrf
        n += 1
    if acc is None or n == 0:
        return
    mean_img = (acc / n)
    mean_img = np.clip(mean_img * 255.0, 0, 255).astype(np.uint8)
    out = report_dir / f"mean_image_{c}.png"
    Image.fromarray(mean_img).save(out)


def save_montage(clean_root: Path, report_dir: Path, c: str, grid: Tuple[int, int] = (5,5)):
    rows, cols = grid
    paths = list_cleaned_images(clean_root).get(c, [])
    paths = sample_paths(paths, rows*cols)
    if not paths:
        return
    images: List[np.ndarray] = []
    for p in paths:
        arr = read_image_rgb(p)
        if arr is None:
            continue
        images.append(arr)
    if not images:
        return
    h, w, _ = images[0].shape
    canvas = np.ones((rows*h, cols*w, 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images[:rows*cols]):
        r = idx // cols
        cidx = idx % cols
        canvas[r*h:(r+1)*h, cidx*w:(cidx+1)*w, :] = img
    out = report_dir / f"montage_{c}.png"
    Image.fromarray(canvas).save(out)


def write_csv_counts_cleaned(clean_root: Path, report_dir: Path):
    counts = counts_cleaned(clean_root)
    out = report_dir / "counts_cleaned.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "count"])
        for c in CLASSES:
            w.writerow([c, counts.get(c, 0)])


def write_csv_counts_split(split_root: Path, report_dir: Path):
    counts = counts_split(split_root)
    out = report_dir / "counts_split.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "count"])
        for split in ("train", "val", "test"):
            for c in CLASSES:
                w.writerow([split, c, counts.get(split, {}).get(c, 0)])


def write_csv_filetypes(clean_root: Path, report_dir: Path):
    dist = filetype_distribution(clean_root)
    out = report_dir / "filetypes.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "ext", "count"])
        for c in CLASSES:
            for ext, cnt in sorted(dist.get(c, {}).items()):
                w.writerow([c, ext, cnt])


def write_csv_channel_stats(clean_root: Path, report_dir: Path, sample_per_class: int):
    stats = compute_channel_stats(clean_root, sample_per_class)
    out = report_dir / "channel_stats.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_images", "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "brightness_mean", "brightness_std"])
        for c in CLASSES:
            s = stats.get(c)
            if not s:
                continue
            w.writerow([
                c,
                s["n_images"],
                f"{s['mean_r']:.4f}", f"{s['mean_g']:.4f}", f"{s['mean_b']:.4f}",
                f"{s['std_r']:.4f}", f"{s['std_g']:.4f}", f"{s['std_b']:.4f}",
                f"{s['brightness_mean']:.4f}", f"{s['brightness_std']:.4f}"
            ])


def main():
    parser = argparse.ArgumentParser(description="EDA for cleaned Extreme/Normal dataset")
    parser.add_argument("--clean-root", type=str, default=str(Path("cleaned_data").resolve()), help="Path to cleaned_data root")
    parser.add_argument("--split-root", type=str, default=str(Path("dataset_split").resolve()), help="Path to dataset_split root")
    parser.add_argument("--report-dir", type=str, default=str(Path("OUTPUT")/"EDA"), help="Directory to write OUTPUT")
    parser.add_argument("--sample-per-class", type=int, default=500, help="Sample size per class for stats and plots (set 0 to use all)")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    split_root = Path(args.split_root)
    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)

    # CSV summaries
    write_csv_counts_cleaned(clean_root, report_dir)
    write_csv_counts_split(split_root, report_dir)
    write_csv_filetypes(clean_root, report_dir)
    write_csv_channel_stats(clean_root, report_dir, args.sample_per_class)

    # Figures
    plot_brightness_hist(clean_root, report_dir, sample_per_class=args.sample_per_class)
    for c in CLASSES:
        save_mean_image(clean_root, report_dir, c, sample_per_class=args.sample_per_class)
        save_montage(clean_root, report_dir, c, grid=(5,5))

    print(f"EDA complete. Reports written to: {report_dir}")


if __name__ == "__main__":
    main()

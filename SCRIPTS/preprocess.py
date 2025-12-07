#!/usr/bin/env python3
"""
Preprocess satellite images for Extreme vs Normal weather classification.
- Deduplicate files by filename rule (any '(' or ')' indicates a duplicate) in source.
- Resize to 224x224, RGB, normalize to [0,1] in-memory, save processed images to cleaned_data/{extreme,normal}.
- Stratified 80/10/10 split into dataset_split/{train,val,test}/{extreme,normal} by copying from cleaned_data.
- Designed for macOS/Linux paths; no GPU dependencies.

Usage:
  python SCRIPTS/preprocess.py [--input INPUT_DIR] [--dry-run]

Defaults:
  INPUT_DIR: If ./dataverse_files exists, use it; else current working directory.
"""

import os
import sys
import argparse
import shutil
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

# Third-party deps
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional
    cv2 = None
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

EXTREME_FOLDERS = {"hurricane", "wildfires", "duststorm"}
NORMAL_FOLDERS = {"convective", "roll"}
# Fallback keyword mapping if flat files are used
EXTREME_KEYWORDS = {"hurricane", "cyclone", "tropical", "wildfire", "fire", "duststorm", "dust"}
NORMAL_KEYWORDS = {"convective", "roll"}

TARGET_SIZE = (224, 224)
CLEAN_ROOT = Path("cleaned_data")
SPLIT_ROOT = Path("dataset_split")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def find_input_dir(cli_input: str | None) -> Path:
    if cli_input:
        return Path(cli_input).resolve()
    cwd = Path.cwd()
    dv = cwd / "dataverse_files"
    return dv if dv.exists() else cwd


def list_all_files(input_dir: Path) -> List[Path]:
    return [p for p in input_dir.rglob("*") if p.is_file()]


def remove_duplicates_in_place(files: List[Path], dry_run: bool = False) -> int:
    dupes = [p for p in files if "(" in p.name or ")" in p.name]
    if not dry_run:
        for p in dupes:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                # best-effort; skip errors
                pass
    return len(dupes)


def map_label(path: Path, base_dir: Path) -> str | None:
    # Prefer folder-based mapping: parent folder relative to base_dir
    try:
        rel = path.relative_to(base_dir)
    except Exception:
        rel = path
    parts = [part.lower() for part in rel.parts]
    # find first known folder in path
    for part in parts:
        if part in EXTREME_FOLDERS:
            return "extreme"
        if part in NORMAL_FOLDERS:
            return "normal"
    # fallback to filename keyword mapping for flat layouts
    name = path.stem.lower()
    if any(k in name for k in EXTREME_KEYWORDS):
        return "extreme"
    if any(k in name for k in NORMAL_KEYWORDS):
        return "normal"
    return None


def ensure_dirs():
    (CLEAN_ROOT / "extreme").mkdir(parents=True, exist_ok=True)
    (CLEAN_ROOT / "normal").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for cls in ("extreme", "normal"):
            (SPLIT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


def read_image_rgb(path: Path) -> np.ndarray | None:
    # Returns RGB float32 array in [0,1]
    try:
        if cv2 is not None:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            # OpenCV loads BGR; convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            return img
        else:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE, resample=Image.BILINEAR)
                arr = np.asarray(im).astype(np.float32) / 255.0
                return arr
    except Exception:
        return None


def deterministic_name(path: Path, base_dir: Path) -> str:
    try:
        rel = str(path.relative_to(base_dir))
    except Exception:
        rel = str(path)
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return f"{h}{path.suffix.lower()}"


def save_image_uint8(rgb01: np.ndarray, out_path: Path) -> None:
    # Persist as PNG/JPG with 0-255 uint8
    arr = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    if cv2 is not None:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
    else:
        Image.fromarray(arr).save(out_path)


def preprocess_to_clean(input_dir: Path, files: List[Path], dry_run: bool = False) -> Tuple[List[Tuple[Path, str]], Dict[str, int]]:
    cleaned: List[Tuple[Path, str]] = []
    counts = {"extreme": 0, "normal": 0, "skipped": 0}
    for p in files:
        if not is_image(p):
            continue
        label = map_label(p, input_dir)
        if label not in ("extreme", "normal"):
            counts["skipped"] += 1
            continue
        img = read_image_rgb(p)
        if img is None:
            counts["skipped"] += 1
            continue
        out_dir = CLEAN_ROOT / label
        out_name = deterministic_name(p, input_dir)
        out_path = out_dir / out_name
        if not dry_run:
            save_image_uint8(img, out_path)
        cleaned.append((out_path, label))
        counts[label] += 1
    return cleaned, counts


def stratified_split_and_copy(cleaned: List[Tuple[Path, str]], dry_run: bool = False) -> Dict[str, Dict[str, int]]:
    # Build arrays
    X = [str(p) for p, _ in cleaned]
    y = [lbl for _, lbl in cleaned]
    if not X:
        return {"train": {"extreme": 0, "normal": 0}, "val": {"extreme": 0, "normal": 0}, "test": {"extreme": 0, "normal": 0}}

    # 80/20 then 50/50 of the 20% to make 10/10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    counts: Dict[str, Dict[str, int]] = {s: {"extreme": 0, "normal": 0} for s in splits}

    for split, (paths, labels) in splits.items():
        for src, lbl in zip(paths, labels):
            dst = SPLIT_ROOT / split / lbl / Path(src).name
            if not dry_run:
                shutil.copy2(src, dst)
            counts[split][lbl] += 1
    return counts


def summarize(total_seen: int, dupes_removed: int, clean_counts: Dict[str, int], split_counts: Dict[str, Dict[str, int]]):
    print("=== Preprocessing Summary ===")
    print(f"Total files seen: {total_seen}")
    print(f"Duplicates removed by name rule: {dupes_removed}")
    print("Cleaned image counts:")
    print(f"  extreme: {clean_counts.get('extreme', 0)}")
    print(f"  normal : {clean_counts.get('normal', 0)}")
    print(f"  skipped: {clean_counts.get('skipped', 0)} (non-image/unmappable/corrupt)")
    print("Split counts:")
    for split in ("train", "val", "test"):
        c = split_counts.get(split, {"extreme": 0, "normal": 0})
        print(f"  {split:5s} -> extreme: {c['extreme']:4d} | normal: {c['normal']:4d}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for Extreme vs Normal classification")
    parser.add_argument("--input", type=str, default=None, help="Input directory (defaults to ./dataverse_files if exists else cwd)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and log without writing any files")
    args = parser.parse_args()

    input_dir = find_input_dir(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found or not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_dirs()

    files = list_all_files(input_dir)
    total_seen = len(files)
    dupes_removed = remove_duplicates_in_place(files, dry_run=args.dry_run)

    # refresh file list if we actually deleted
    if not args.dry_run:
        files = list_all_files(input_dir)

    cleaned, clean_counts = preprocess_to_clean(input_dir, files, dry_run=args.dry_run)

    split_counts = stratified_split_and_copy(cleaned, dry_run=args.dry_run)

    summarize(total_seen, dupes_removed, clean_counts, split_counts)


if __name__ == "__main__":
    main()

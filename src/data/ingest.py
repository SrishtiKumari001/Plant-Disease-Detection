from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

from src.utils import ensure_dir, project_root


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_pair(raw: str) -> Tuple[Path, str]:
    if "::" not in raw:
        raise ValueError(f"Invalid pair '{raw}'. Expected format: <image_path>::<label>")
    path_str, label = raw.split("::", maxsplit=1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid pair '{raw}'. Label cannot be empty.")
    return Path(path_str.strip()).expanduser().resolve(), label


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def unique_target_path(class_dir: Path, src: Path, prefix: str) -> Path:
    safe_stem = sanitize_name(src.stem)
    safe_prefix = sanitize_name(prefix) if prefix else "added"
    candidate = class_dir / f"{safe_prefix}_{safe_stem}{src.suffix.lower()}"
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = class_dir / f"{safe_prefix}_{safe_stem}_{idx}{src.suffix.lower()}"
        if not candidate.exists():
            return candidate
        idx += 1


def ingest_images(
    pairs: List[Tuple[Path, str]],
    data_root: Path,
    mode: str,
    prefix: str,
    dry_run: bool,
) -> List[Tuple[Path, Path, str]]:
    actions: List[Tuple[Path, Path, str]] = []

    for src, label in pairs:
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Image not found: {src}")
        if src.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Unsupported image extension for {src}. Allowed: {sorted(IMG_EXTS)}")

        class_dir = ensure_dir(data_root / label)
        dst = unique_target_path(class_dir, src, prefix=prefix)
        actions.append((src, dst, label))

    for src, dst, label in actions:
        print(f"[{mode}] {src} -> {dst} (label={label})")
        if dry_run:
            continue

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "move":
            shutil.move(str(src), str(dst))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return actions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest labeled external images into PlantVillage class folders.")
    p.add_argument(
        "--pair",
        action="append",
        required=True,
        help="One labeled image in format: <image_path>::<label>. Repeat --pair for multiple images.",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(project_root() / "data" / "raw" / "PlantVillage"),
        help="Dataset class-folder root.",
    )
    p.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move source files into class folders.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="real",
        help="Filename prefix used when creating destination names.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without writing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    ensure_dir(data_root)

    pairs = [parse_pair(raw) for raw in args.pair]
    actions = ingest_images(
        pairs=pairs,
        data_root=data_root,
        mode=args.mode,
        prefix=args.prefix,
        dry_run=bool(args.dry_run),
    )

    print("\nSummary")
    print(f"Data root: {data_root}")
    print(f"Images processed: {len(actions)}")
    if args.dry_run:
        print("Dry run only. No files were written.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.transforms import build_eval_transforms
from src.models.cnn_baseline import CNNBaseline
from src.models.transfer import build_model
from src.utils import get_device, load_json, project_root


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_names() -> Optional[List[str]]:
    root = project_root()
    label_map_path = root / "data" / "splits" / "label_map.json"
    if not label_map_path.exists():
        return None

    label_map = load_json(label_map_path)
    inv = [None] * (max(label_map.values()) + 1)
    for name, idx in label_map.items():
        inv[idx] = name
    return [name if name is not None else f"class_{idx}" for idx, name in enumerate(inv)]


def infer_num_classes(cfg: Dict[str, Any], fallback: Optional[int] = None) -> int:
    root = project_root()
    label_names = load_label_names()
    if label_names is not None:
        return len(label_names)

    split_csv = root / cfg["data"]["split_csv"]
    if split_csv.exists():
        import pandas as pd

        df = pd.read_csv(split_csv)
        return int(df["label_idx"].nunique())

    if fallback is not None:
        return int(fallback)

    raise FileNotFoundError("Could not infer num_classes. Generate data/splits/split.csv first.")


def build_from_config(cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    name = cfg["model"]["name"].lower().strip()
    pretrained = bool(cfg["model"].get("pretrained", True))
    dropout = float(cfg["model"].get("dropout", 0.0))

    if name in {"baseline_cnn", "cnn_baseline"}:
        return CNNBaseline(num_classes=num_classes, dropout=dropout)

    return build_model(name=name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def default_checkpoint_path(cfg: Dict[str, Any]) -> Path:
    root = project_root()
    exp_name = cfg["experiment"]["name"]
    checkpoints_dir = root / "reports" / "checkpoints"

    exact_best = checkpoints_dir / f"{exp_name}_best.pt"
    if exact_best.exists():
        return exact_best

    model_name = str(cfg["model"]["name"]).lower().strip()
    model_matches = sorted(
        checkpoints_dir.glob(f"{model_name}*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if model_matches:
        return model_matches[0]

    return exact_best


def resolve_checkpoint_path(cfg: Dict[str, Any], checkpoint_path: Optional[str] = None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path).expanduser().resolve()
    return default_checkpoint_path(cfg)


def load_model_for_inference(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], List[str], torch.device, Path]:
    cfg = load_config(config_path)
    prefer_mps = bool(cfg.get("device", {}).get("prefer_mps", True))
    device = device or get_device(prefer_mps=prefer_mps)

    ckpt_path = resolve_checkpoint_path(cfg, checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint missing key 'model_state'.")

    num_classes = infer_num_classes(cfg, fallback=ckpt.get("num_classes"))
    model = build_from_config(cfg, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    label_names = load_label_names() or [f"class_{idx}" for idx in range(num_classes)]
    if len(label_names) != num_classes:
        label_names = [f"class_{idx}" for idx in range(num_classes)]

    return model, cfg, label_names, device, ckpt_path


def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    transform = build_eval_transforms(img_size=img_size)
    return transform(image.convert("RGB")).unsqueeze(0)


def _tta_preprocess_variants(image: Image.Image, img_size: int) -> List[torch.Tensor]:
    """
    Build a small set of deterministic test-time augmentation views.

    The goal is to smooth out single-crop mistakes on leaf photos by averaging
    predictions over a few plausible views rather than changing the training path.
    """
    rgb = image.convert("RGB")
    base_transform = build_eval_transforms(img_size=img_size)
    variants = [base_transform(rgb)]

    flipped = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    variants.append(base_transform(flipped))

    # Small center-crop jitter around the evaluation geometry.
    w, h = rgb.size
    crop_pad = int(round(min(w, h) * 0.03))
    if crop_pad > 0 and w > 2 * crop_pad and h > 2 * crop_pad:
        crops = [
            (0, 0, w - crop_pad, h - crop_pad),
            (crop_pad, 0, w, h - crop_pad),
            (0, crop_pad, w - crop_pad, h),
            (crop_pad, crop_pad, w, h),
        ]
        for box in crops:
            variants.append(base_transform(rgb.crop(box)))

    return variants


@torch.no_grad()
def predict_image_tta(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    img_size: int,
    label_names: Sequence[str],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Predict with a small TTA ensemble of deterministic crops/flips.
    """
    views = _tta_preprocess_variants(image, img_size=img_size)
    batch = torch.cat(views, dim=0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).mean(dim=0).detach().cpu().numpy()

    top_k = max(1, min(int(top_k), probs.shape[0]))
    top_indices = np.argsort(probs)[-top_k:][::-1]

    predictions = [
        {
            "rank": rank + 1,
            "class_index": int(idx),
            "label": label_names[int(idx)],
            "confidence": float(probs[int(idx)]),
        }
        for rank, idx in enumerate(top_indices)
    ]

    best = predictions[0]
    return {
        "predictions": predictions,
        "top_prediction": best,
        "probabilities": probs,
        "num_views": len(views),
        "tta_enabled": True,
    }


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    img_size: int,
    label_names: Sequence[str],
    top_k: int = 3,
    use_tta: bool = False,
) -> Dict[str, Any]:
    if use_tta:
        return predict_image_tta(
            model=model,
            image=image,
            device=device,
            img_size=img_size,
            label_names=label_names,
            top_k=top_k,
        )

    x = preprocess_image(image, img_size=img_size).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_k = max(1, min(int(top_k), probs.shape[0]))
    top_indices = np.argsort(probs)[-top_k:][::-1]

    predictions = [
        {
            "rank": rank + 1,
            "class_index": int(idx),
            "label": label_names[int(idx)],
            "confidence": float(probs[int(idx)]),
        }
        for rank, idx in enumerate(top_indices)
    ]

    best = predictions[0]
    return {
        "predictions": predictions,
        "top_prediction": best,
        "probabilities": probs,
    }
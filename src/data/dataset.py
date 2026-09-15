# src/data/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import json

from src.utils import project_root


@dataclass(frozen=True)
class Sample:
    """
    One row from split.csv
    """
    filepath: str   # relative path from project root (as stored in split.csv)
    label: str
    label_idx: int
    split: str      # 'train' / 'val' / 'test'


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset for PlantVillage-style data using a split.csv file.

    Expected CSV columns:
      - filepath (relative path from project root)
      - label (class name string)
      - label_idx (int)
      - split ('train'/'val'/'test')
    """

    def __init__(
        self,
        split: str,
        csv_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Parameters:
        - split: 'train' / 'val' / 'test'
        - csv_path: path to data/splits/split.csv (default uses project structure)
        - transform: torchvision transforms pipeline (train or eval)
        """
        split = split.lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of {{train, val, test}}, got: {split}")

        self.split = split
        self.transform = transform

        root = project_root()
        if csv_path is None:
            csv_path = str(root / "data" / "splits" / "split.csv")

        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"split.csv not found at: {self.csv_path}\n"
                f"Did you run: python -m src.data.split ?"
            )

        df = pd.read_csv(self.csv_path)
        required_cols = {"filepath", "label", "label_idx", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"split.csv is missing columns: {missing}")

        df = df[df["split"] == self.split].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No rows found for split='{self.split}' in {self.csv_path}")

        # Load label_map.json if present and remove tomato classes
        label_map_path = root / "data" / "splits" / "label_map.json"
        if label_map_path.exists():
            try:
                with open(label_map_path, "r", encoding="utf-8") as fh:
                    orig_map = json.load(fh)
            except Exception:
                orig_map = {}
        else:
            orig_map = {}

        # Build a filtered label -> idx mapping that excludes any label starting with 'Tomato'
        if orig_map:
            filtered_labels = [k for k in orig_map.keys() if not k.lower().startswith("tomato")]
            label_to_idx = {label: i for i, label in enumerate(filtered_labels)}
        else:
            unique_labels = sorted(df["label"].unique())
            filtered_labels = [l for l in unique_labels if not l.lower().startswith("tomato")]
            label_to_idx = {label: i for i, label in enumerate(filtered_labels)}

        # Filter out rows with tomato labels
        df = df[df["label"].isin(label_to_idx)].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("No rows left after filtering tomato classes; check your splits and label_map.json")

        # Convert dataframe rows into Sample objects using the new label indices
        self.samples = [
            Sample(
                filepath=str(row["filepath"]),
                label=str(row["label"]),
                label_idx=int(label_to_idx[str(row["label"])]),
                split=str(row["split"]),
            )
            for _, row in df.iterrows()
        ]

        self.root = root  # project root path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        """
        Returns:
          image_tensor, label_idx
        """
        s = self.samples[idx]
        img_path = (self.root / s.filepath).resolve()

        if not img_path.exists():
            parts = Path(s.filepath).parts
            if len(parts) >= 3 and parts[0] == "data" and parts[1] == "raw" and parts[2] == "PlantVillage":
                alt_path = (self.root / "data" / "raw" / "PlantVillage" / "PlantVillage" / Path(*parts[3:])).resolve()
                if alt_path.exists():
                    img_path = alt_path

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # force RGB to avoid grayscale/alpha issues
            if self.transform is not None:
                img = self.transform(img)

        return img, s.label_idx

    def get_class_name(self, idx: int) -> str:
        """
        Convenience: return the label string for a given dataset index.
        """
        return self.samples[idx].label

    def get_filepath(self, idx: int) -> str:
        return str(self.samples[idx].filepath)


"""Preprocessing helpers for multimodal difficult-airway prediction.

The functions in this module are deliberately independent of the model so that
the same quality-control and sampling policy can be reused by image, video and
multimodal experiments.  Resampling is applied to the training split only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler


def class_counts(labels) -> dict[int, int]:
    """Return deterministic class counts for logging and QC reports."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    return {int(label): int(count) for label, count in sorted(Counter(values).items())}


def make_weighted_sampler(
    labels,
    *,
    replacement: bool = True,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Create an inverse-frequency sampler for a binary/multiclass target.

    ``WeightedRandomSampler`` samples indices rather than duplicating media on
    disk, so the validation split remains untouched and every modality stays
    aligned.  By default one epoch has the same number of draws as the original
    training set.
    """
    values = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.long)
    if values.numel() == 0:
        raise ValueError("Cannot build a sampler for an empty label array")
    unique, counts = torch.unique(values, return_counts=True)
    if unique.numel() < 2:
        # A one-class fold is valid for feature extraction; sampling it is just
        # uniform and avoids division-by-zero.
        weights = torch.ones(values.numel(), dtype=torch.double)
    else:
        inverse = torch.zeros(int(unique.max().item()) + 1, dtype=torch.double)
        inverse[unique] = 1.0 / counts.to(torch.double)
        weights = inverse[values]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples or values.numel()),
        replacement=replacement,
    )


def validate_media_file(path: str | Path, media_type: str) -> tuple[bool, str]:
    """Perform inexpensive, deterministic media quality control."""
    candidate = Path(path)
    if not candidate.is_file():
        return False, "missing_file"
    try:
        if media_type == "image":
            with Image.open(candidate) as image:
                image.verify()
        elif media_type == "video":
            # Header inspection catches truncated files without decoding every
            # frame during cross-validation startup.
            import cv2

            capture = cv2.VideoCapture(str(candidate))
            opened = bool(capture.isOpened())
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            capture.release()
            if not opened or frame_count <= 0:
                return False, "unreadable_video"
        else:
            raise ValueError(f"Unknown media_type: {media_type}")
    except Exception as exc:  # corrupt image/video or unsupported codec
        return False, f"unreadable_{media_type}:{type(exc).__name__}"
    return True, "ok"


def quality_control_table(
    table: pd.DataFrame,
    *,
    image_columns: list[str] | None = None,
    video_columns: list[str] | None = None,
    drop_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check paths and media headers, returning ``(clean, report)``."""
    image_columns = image_columns or []
    video_columns = video_columns or []
    records = []
    keep = []
    for row_index, row in table.reset_index(drop=True).iterrows():
        reasons = []
        for index, column in enumerate(image_columns, start=1):
            valid, reason = validate_media_file(row[f"image_path_{index}"], "image")
            if not valid:
                reasons.append(f"{column}:{reason}")
        for index, column in enumerate(video_columns, start=1):
            valid, reason = validate_media_file(row[f"video_path_{index}"], "video")
            if not valid:
                reasons.append(f"{column}:{reason}")
        keep.append(not reasons)
        records.append({"row_index": row_index, "valid": not reasons, "reason": ";".join(reasons)})
    report = pd.DataFrame(records)
    if drop_invalid:
        clean = table.reset_index(drop=True).loc[np.asarray(keep)].reset_index(drop=True)
    else:
        clean = table.reset_index(drop=True)
    return clean, report

"""Quantitative matching and panorama-quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class PairMetrics:
    detector: str
    n_keypoints_a: int
    n_keypoints_b: int
    n_initial_matches: int
    n_inliers: int
    inlier_ratio: float
    mean_reproj_error: float
    detect_s: float
    match_s: float
    ransac_s: float
    total_s: float
    overlap_ssim: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def inlier_ratio(n_inliers: int, n_matches: int) -> float:
    if n_matches <= 0:
        return 0.0
    return n_inliers / float(n_matches)


def overlap_ssim(warped_a: np.ndarray, warped_b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    overlap = (mask_a > 0) & (mask_b > 0)
    if overlap.sum() < 64:
        return float("nan")
    gray_a = cv2.cvtColor(warped_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    roi_a = gray_a[y0:y1, x0:x1]
    roi_b = gray_b[y0:y1, x0:x1]
    if roi_a.size < 64:
        return float("nan")
    try:
        value = ssim(roi_a, roi_b, data_range=255, win_size=min(7, min(roi_a.shape) | 1))
    except ValueError:
        value = float("nan")
    return float(value)

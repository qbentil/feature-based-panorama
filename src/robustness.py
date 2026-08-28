"""Synthetic rotation, scale, and illumination changes for robustness tests."""

from __future__ import annotations

import cv2
import numpy as np


def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(0, 0, 0))


def scale(image: np.ndarray, factor: float) -> np.ndarray:
    factor = max(0.1, float(factor))
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)


def illuminate(image: np.ndarray, gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=gain, beta=bias)

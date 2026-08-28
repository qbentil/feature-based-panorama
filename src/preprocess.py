"""Image preparation before feature detection."""

from __future__ import annotations

import cv2
import numpy as np


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_side:
        return image
    scale = max_side / float(long_edge)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize is None or ksize < 3:
        return image
    k = int(ksize) | 1
    return cv2.GaussianBlur(image, (k, k), 0)


def prepare(
    image_bgr: np.ndarray,
    max_side: int = 1200,
    blur_ksize: int = 3,
    use_clahe: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (colour BGR, gray-for-features) after resize / denoise / CLAHE."""
    colour = resize_max_side(image_bgr, max_side)
    colour = denoise(colour, blur_ksize)
    gray = to_gray(colour)
    if use_clahe:
        gray = apply_clahe(gray)
    return colour, gray

"""RANSAC homography estimation and perspective warping."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_homography(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    ransac_thresh: float = 5.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (H, inlier_mask) mapping src points to dst points."""
    if len(pts_src) < 4 or len(pts_dst) < 4:
        return None, None
    H, mask = cv2.findHomography(
        pts_src.reshape(-1, 1, 2),
        pts_dst.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    return H, mask


def mean_reprojection_error(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    H: np.ndarray | None,
    inlier_mask: np.ndarray | None,
) -> float:
    if H is None or inlier_mask is None or len(pts_src) == 0:
        return float("nan")
    mask = inlier_mask.ravel().astype(bool)
    if not np.any(mask):
        return float("nan")
    src = pts_src[mask].reshape(-1, 1, 2)
    dst = pts_dst[mask].reshape(-1, 2)
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(projected - dst, axis=1)
    return float(np.mean(err))


def invert_homography(H: np.ndarray) -> np.ndarray:
    return np.linalg.inv(H)


def compose_to_reference(pair_hs: list[np.ndarray], n_images: int, ref_index: int) -> list[np.ndarray]:
    """pair_hs[i] maps image i -> image i+1. Return H_i mapping each image to the reference."""
    if len(pair_hs) != n_images - 1:
        raise ValueError("Need one homography per consecutive pair.")
    Hs = [np.eye(3, dtype=np.float64) for _ in range(n_images)]
    for i in range(ref_index - 1, -1, -1):
        Hs[i] = Hs[i + 1] @ pair_hs[i]
    for i in range(ref_index + 1, n_images):
        Hs[i] = Hs[i - 1] @ invert_homography(pair_hs[i - 1])
    return Hs


def warped_corners(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H)


def canvas_from_homographies(
    images: list[np.ndarray],
    Hs: list[np.ndarray],
    max_dim: int = 8000,
) -> tuple[list[np.ndarray], int, int]:
    all_corners = [warped_corners(img, H) for img, H in zip(images, Hs)]
    stacked = np.concatenate(all_corners, axis=0).reshape(-1, 2)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    tx, ty = -xmin, -ymin
    translate = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    Hs_t = [translate @ H for H in Hs]
    width = int(np.ceil(xmax - xmin))
    height = int(np.ceil(ymax - ymin))
    width = max(1, min(width, max_dim))
    height = max(1, min(height, max_dim))
    return Hs_t, width, height


def warp_image(image: np.ndarray, H: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    warped = cv2.warpPerspective(image, H, (width, height))
    mask = cv2.warpPerspective(
        np.ones(image.shape[:2], dtype=np.uint8) * 255,
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    return warped, mask

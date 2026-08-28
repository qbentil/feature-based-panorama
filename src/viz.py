"""Draw keypoints, matches, and alignment overlays (BGR in, RGB out for display)."""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_keypoints(image_bgr: np.ndarray, keypoints: list[cv2.KeyPoint], max_draw: int = 400) -> np.ndarray:
    kps = keypoints[:max_draw]
    canvas = cv2.drawKeypoints(
        image_bgr,
        kps,
        None,
        color=(0, 200, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return bgr_to_rgb(canvas)


def _side_by_side(img_a: np.ndarray, img_b: np.ndarray) -> tuple[np.ndarray, int]:
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = max(ha, hb)
    canvas = np.zeros((h, wa + wb, 3), dtype=np.uint8)
    canvas[:ha, :wa] = img_a
    canvas[:hb, wa : wa + wb] = img_b
    return canvas, wa


def draw_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 250,
    inliers_only: bool = False,
) -> np.ndarray:
    canvas, offset = _side_by_side(img_a, img_b)
    n = len(pts_a)
    if n == 0:
        return bgr_to_rgb(canvas)
    if inlier_mask is None:
        keep = np.ones(n, dtype=bool)
    else:
        keep = inlier_mask.ravel().astype(bool)
        if not inliers_only:
            # Draw outliers first in red, inliers on top in green.
            _draw_lines(canvas, pts_a, pts_b, offset, ~keep, (40, 40, 220), max_draw)
        _draw_lines(canvas, pts_a, pts_b, offset, keep, (40, 200, 60), max_draw)
        return bgr_to_rgb(canvas)
    _draw_lines(canvas, pts_a, pts_b, offset, keep, (40, 180, 255), max_draw)
    return bgr_to_rgb(canvas)


def _draw_lines(canvas, pts_a, pts_b, offset, keep, color, max_draw) -> None:
    idx = np.where(keep)[0][:max_draw]
    for i in idx:
        pa = (int(pts_a[i, 0]), int(pts_a[i, 1]))
        pb = (int(pts_b[i, 0]) + offset, int(pts_b[i, 1]))
        cv2.circle(canvas, pa, 4, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pb, 4, color, 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, pa, pb, color, 1, lineType=cv2.LINE_AA)


def warp_overlay(img_a: np.ndarray, img_b: np.ndarray, H_a_to_b: np.ndarray) -> np.ndarray:
    """Warp A into B's frame and blend so alignment is visible."""
    h, w = img_b.shape[:2]
    warped_a = cv2.warpPerspective(img_a, H_a_to_b, (w, h))
    overlay = cv2.addWeighted(img_b, 0.55, warped_a, 0.45, 0)
    return bgr_to_rgb(overlay)

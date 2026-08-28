"""Multi-image panorama assembly with distance-transform feathering."""

from __future__ import annotations

import cv2
import numpy as np

from src.homography import canvas_from_homographies, compose_to_reference, warp_image


def feather_blend(warped: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(warped[0], dtype=np.float32)
    weight = np.zeros(warped[0].shape[:2], dtype=np.float32)
    for img, mask in zip(warped, masks):
        binary = (mask > 0).astype(np.uint8)
        if binary.max() == 0:
            continue
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        w = dist.astype(np.float32)
        acc += img.astype(np.float32) * w[..., None]
        weight += w
    denom = np.maximum(weight[..., None], 1e-6)
    blended = acc / denom
    empty = weight < 1e-6
    blended[empty] = 0
    return np.clip(blended, 0, 255).astype(np.uint8)


def crop_black_borders(panorama: np.ndarray, margin: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) if panorama.ndim == 3 else panorama
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return panorama
    x, y, w, h = cv2.boundingRect(coords)
    x = max(0, x + margin)
    y = max(0, y + margin)
    w = max(1, w - 2 * margin)
    h = max(1, h - 2 * margin)
    return panorama[y : y + h, x : x + w]


def assemble_panorama(
    images: list[np.ndarray],
    pair_hs: list[np.ndarray],
    ref_index: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Warp every image into the middle-image frame and blend.

    Returns panorama, translated homographies, warped images, warped masks.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images to stitch.")
    if ref_index is None:
        ref_index = n // 2
    Hs = compose_to_reference(pair_hs, n, ref_index)
    Hs_t, width, height = canvas_from_homographies(images, Hs)
    warped = []
    masks = []
    for img, H in zip(images, Hs_t):
        w_img, w_mask = warp_image(img, H, width, height)
        warped.append(w_img)
        masks.append(w_mask)
    panorama = feather_blend(warped, masks)
    panorama = crop_black_borders(panorama)
    return panorama, Hs_t, warped, masks

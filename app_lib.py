"""Shared Streamlit helpers. Keep pages as thin UI scripts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.paths import list_images, list_scenes
from src.pipeline import PairResult, StitchResult, match_pair, stitch_images
from src.robustness import illuminate, rotate, scale
from src.viz import bgr_to_rgb

THUMB_FILL = (235, 230, 220)
THUMB_SIZE = 200
DISPLAY_MAX_SIDE = 900


def scene_options() -> dict[str, Path]:
    return {p.name.replace("_", " "): p for p in list_scenes()}


def load_images_from_dir(folder: Path) -> list[np.ndarray]:
    images = []
    for path in list_images(folder):
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    return images


def decode_uploads(files) -> list[np.ndarray]:
    images = []
    for uploaded in files:
        buf = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images


@st.cache_data(max_entries=8, show_spinner=False)
def cached_stitch(
    image_bytes: tuple[bytes, ...],
    detector: str,
    ratio: float,
    ransac_thresh: float,
    max_side: int,
    blur_ksize: int,
    use_clahe: bool,
) -> StitchResult:
    images = []
    for raw in image_bytes:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Could not decode one of the input images.")
        images.append(img)
    return stitch_images(
        images,
        detector=detector,
        ratio=ratio,
        ransac_thresh=ransac_thresh,
        max_side=max_side,
        blur_ksize=blur_ksize,
        use_clahe=use_clahe,
    )


@st.cache_data(max_entries=16, show_spinner=False)
def cached_pair(
    a_bytes: bytes,
    b_bytes: bytes,
    detector: str,
    ratio: float,
    ransac_thresh: float,
    max_side: int,
    blur_ksize: int,
    use_clahe: bool,
) -> PairResult:
    a = cv2.imdecode(np.frombuffer(a_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.frombuffer(b_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return match_pair(
        a,
        b,
        detector=detector,
        ratio=ratio,
        ransac_thresh=ransac_thresh,
        max_side=max_side,
        blur_ksize=blur_ksize,
        use_clahe=use_clahe,
    )


def encode_bgr(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    return buf.tobytes()


def transform_bgr(image: np.ndarray, kind: str, amount: float) -> np.ndarray:
    if kind == "rotation":
        return rotate(image, amount)
    if kind == "scale":
        return scale(image, amount)
    if kind == "illumination":
        return illuminate(image, gain=amount, bias=0.0)
    raise ValueError(kind)


def metric_row(metrics, extra: dict | None = None) -> None:
    cols = st.columns(4)
    cols[0].metric("Keypoints (left / right)", f"{metrics.n_keypoints_a} / {metrics.n_keypoints_b}")
    cols[1].metric("Initial matches", f"{metrics.n_initial_matches}")
    cols[2].metric("RANSAC inliers", f"{metrics.n_inliers}")
    cols[3].metric("Inlier ratio", f"{metrics.inlier_ratio:.2%}")
    cols2 = st.columns(4)
    reproj = metrics.mean_reproj_error
    cols2[0].metric("Reprojection error", "—" if not np.isfinite(reproj) else f"{reproj:.2f} px")
    ssim = metrics.overlap_ssim
    cols2[1].metric("Overlap SSIM", "—" if not np.isfinite(ssim) else f"{ssim:.3f}")
    cols2[2].metric("Detect + match", f"{metrics.detect_s + metrics.match_s:.2f} s")
    cols2[3].metric("Total pair time", f"{metrics.total_s:.2f} s")
    if extra:
        for label, value in extra.items():
            st.caption(f"{label}: {value}")


def square_thumb(image_bgr: np.ndarray, size: int = THUMB_SIZE) -> np.ndarray:
    """Letterbox a BGR image into a fixed square RGB tile for the UI."""
    rgb = bgr_to_rgb(image_bgr) if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), THUMB_FILL, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def fit_display(image: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> np.ndarray:
    """Shrink a display image so a huge panorama cannot stretch the page."""
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_side:
        return rgb
    scale = max_side / float(long_edge)
    return cv2.resize(
        rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def show_thumbs(images_bgr: list[np.ndarray], max_n: int = 5) -> None:
    n = min(len(images_bgr), max_n)
    cols = st.columns(n)
    for col, img in zip(cols, images_bgr[:n]):
        col.image(square_thumb(img), caption=f"{img.shape[1]}×{img.shape[0]}")
    if len(images_bgr) > max_n:
        st.caption(f"{len(images_bgr)} images loaded.")


def show_rgb(image: np.ndarray, caption: str = "") -> None:
    rgb = image if image.ndim == 3 and image.shape[2] == 3 else bgr_to_rgb(image)
    st.image(fit_display(rgb), caption=caption)

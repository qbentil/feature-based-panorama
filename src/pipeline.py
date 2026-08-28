"""End-to-end pair matching and multi-image stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from src.evaluate import PairMetrics, inlier_ratio, overlap_ssim
from src.features import detect_and_describe
from src.homography import estimate_homography, invert_homography, mean_reprojection_error, warp_image
from src.matching import match_point_arrays, ratio_matches
from src.preprocess import prepare
from src.stitch import assemble_panorama
from src.viz import bgr_to_rgb, draw_keypoints, draw_matches, warp_overlay


@dataclass
class PairResult:
    colour_a: np.ndarray
    colour_b: np.ndarray
    metrics: PairMetrics
    keypoints_a_vis: np.ndarray
    keypoints_b_vis: np.ndarray
    matches_raw_vis: np.ndarray
    matches_inlier_vis: np.ndarray
    overlay_vis: np.ndarray | None
    H_a_to_b: np.ndarray | None
    inlier_mask: np.ndarray | None
    pts_a: np.ndarray
    pts_b: np.ndarray


@dataclass
class StitchResult:
    panorama_bgr: np.ndarray
    panorama_rgb: np.ndarray
    pair_results: list[PairResult] = field(default_factory=list)
    detector: str = "SIFT"
    ref_index: int = 0
    stitch_s: float = 0.0
    mean_inlier_ratio: float = 0.0
    mean_reproj_error: float = float("nan")
    overlap_ssim: float = float("nan")
    colours: list[np.ndarray] = field(default_factory=list)


def match_pair(
    image_a_bgr: np.ndarray,
    image_b_bgr: np.ndarray,
    detector: str = "SIFT",
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    max_side: int = 1200,
    blur_ksize: int = 3,
    use_clahe: bool = True,
    n_features: int = 2000,
) -> PairResult:
    t0 = time.perf_counter()
    colour_a, gray_a = prepare(image_a_bgr, max_side=max_side, blur_ksize=blur_ksize, use_clahe=use_clahe)
    colour_b, gray_b = prepare(image_b_bgr, max_side=max_side, blur_ksize=blur_ksize, use_clahe=use_clahe)

    t_detect0 = time.perf_counter()
    kps_a, desc_a = detect_and_describe(gray_a, detector, n_features=n_features)
    kps_b, desc_b = detect_and_describe(gray_b, detector, n_features=n_features)
    detect_s = time.perf_counter() - t_detect0

    t_match0 = time.perf_counter()
    matches = ratio_matches(desc_a, desc_b, detector, ratio=ratio)
    pts_a, pts_b = match_point_arrays(kps_a, kps_b, matches)
    match_s = time.perf_counter() - t_match0

    t_ransac0 = time.perf_counter()
    H, mask = estimate_homography(pts_a, pts_b, ransac_thresh=ransac_thresh)
    ransac_s = time.perf_counter() - t_ransac0

    n_inliers = int(mask.ravel().sum()) if mask is not None else 0
    reproj = mean_reprojection_error(pts_a, pts_b, H, mask)
    total_s = time.perf_counter() - t0

    ssim_value = float("nan")
    overlay = None
    if H is not None:
        h, w = colour_b.shape[:2]
        warped_a, mask_a = warp_image(colour_a, H, w, h)
        mask_b = np.ones(colour_b.shape[:2], dtype=np.uint8) * 255
        ssim_value = overlap_ssim(warped_a, colour_b, mask_a, mask_b)
        overlay = warp_overlay(colour_a, colour_b, H)

    metrics = PairMetrics(
        detector=detector.upper(),
        n_keypoints_a=len(kps_a),
        n_keypoints_b=len(kps_b),
        n_initial_matches=len(matches),
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio(n_inliers, len(matches)),
        mean_reproj_error=reproj,
        detect_s=detect_s,
        match_s=match_s,
        ransac_s=ransac_s,
        total_s=total_s,
        overlap_ssim=ssim_value,
    )
    return PairResult(
        colour_a=colour_a,
        colour_b=colour_b,
        metrics=metrics,
        keypoints_a_vis=draw_keypoints(colour_a, kps_a),
        keypoints_b_vis=draw_keypoints(colour_b, kps_b),
        matches_raw_vis=draw_matches(colour_a, colour_b, pts_a, pts_b, mask, inliers_only=False),
        matches_inlier_vis=draw_matches(colour_a, colour_b, pts_a, pts_b, mask, inliers_only=True),
        overlay_vis=overlay,
        H_a_to_b=H,
        inlier_mask=mask,
        pts_a=pts_a,
        pts_b=pts_b,
    )


def stitch_images(
    images_bgr: list[np.ndarray],
    detector: str = "SIFT",
    ratio: float = 0.75,
    ransac_thresh: float = 5.0,
    max_side: int = 1200,
    blur_ksize: int = 3,
    use_clahe: bool = True,
    n_features: int = 2000,
) -> StitchResult:
    if len(images_bgr) < 2:
        raise ValueError("Need at least two images to stitch.")

    pair_results: list[PairResult] = []
    pair_hs: list[np.ndarray] = []

    t0 = time.perf_counter()
    for i in range(len(images_bgr) - 1):
        pair = match_pair(
            images_bgr[i],
            images_bgr[i + 1],
            detector=detector,
            ratio=ratio,
            ransac_thresh=ransac_thresh,
            max_side=max_side,
            blur_ksize=blur_ksize,
            use_clahe=use_clahe,
            n_features=n_features,
        )
        if pair.H_a_to_b is None:
            raise RuntimeError(
                f"Homography failed between images {i + 1} and {i + 2}. "
                "Need more overlap or a different detector."
            )
        pair_results.append(pair)
        pair_hs.append(pair.H_a_to_b)

    colours = [prepare(img, max_side=max_side, blur_ksize=blur_ksize, use_clahe=use_clahe)[0] for img in images_bgr]
    ref_index = len(colours) // 2
    panorama, _, _, _ = assemble_panorama(colours, pair_hs, ref_index=ref_index)
    stitch_s = time.perf_counter() - t0

    ratios = [p.metrics.inlier_ratio for p in pair_results]
    errors = [p.metrics.mean_reproj_error for p in pair_results if np.isfinite(p.metrics.mean_reproj_error)]
    ssims = [p.metrics.overlap_ssim for p in pair_results if np.isfinite(p.metrics.overlap_ssim)]
    return StitchResult(
        panorama_bgr=panorama,
        panorama_rgb=bgr_to_rgb(panorama),
        pair_results=pair_results,
        detector=detector.upper(),
        ref_index=ref_index,
        stitch_s=stitch_s,
        mean_inlier_ratio=float(np.mean(ratios)) if ratios else 0.0,
        mean_reproj_error=float(np.mean(errors)) if errors else float("nan"),
        overlap_ssim=float(np.mean(ssims)) if ssims else float("nan"),
        colours=colours,
    )


# Keep invert available for callers that warp the other direction.
__all__ = ["PairResult", "StitchResult", "match_pair", "stitch_images", "invert_homography"]

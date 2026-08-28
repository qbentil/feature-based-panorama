#!/usr/bin/env python3
"""Run the exam comparison table and robustness figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import DATA_SCENES, FIGURES_DIR, RESULTS_DIR, list_images
from src.pipeline import match_pair, stitch_images
from src.robustness import illuminate, rotate, scale
from src.viz import bgr_to_rgb


def load_folder(name: str) -> list[np.ndarray]:
    folder = DATA_SCENES / name
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    images = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        images.append(img)
    return images


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_table(path: Path, rows: list[dict]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def run_detector_comparison() -> pd.DataFrame:
    images = load_folder("scene_main")
    rows = []
    for detector in ("SIFT", "ORB"):
        result = stitch_images(images, detector=detector)
        save_rgb(FIGURES_DIR / f"panorama_{detector.lower()}.png", result.panorama_rgb)
        # Representative pair: middle two frames
        mid = result.pair_results[len(result.pair_results) // 2]
        save_rgb(FIGURES_DIR / f"keypoints_{detector.lower()}_a.png", mid.keypoints_a_vis)
        save_rgb(FIGURES_DIR / f"keypoints_{detector.lower()}_b.png", mid.keypoints_b_vis)
        save_rgb(FIGURES_DIR / f"matches_before_{detector.lower()}.png", mid.matches_raw_vis)
        save_rgb(FIGURES_DIR / f"matches_after_{detector.lower()}.png", mid.matches_inlier_vis)
        if mid.overlay_vis is not None:
            save_rgb(FIGURES_DIR / f"warp_overlay_{detector.lower()}.png", mid.overlay_vis)

        # Pair-level rows (exam table is per matching experiment)
        for idx, pair in enumerate(result.pair_results):
            m = pair.metrics
            rows.append(
                {
                    "detector": detector,
                    "pair": f"{idx + 1}-{idx + 2}",
                    "n_keypoints_a": m.n_keypoints_a,
                    "n_keypoints_b": m.n_keypoints_b,
                    "n_initial_matches": m.n_initial_matches,
                    "n_inliers": m.n_inliers,
                    "inlier_ratio": round(m.inlier_ratio, 4),
                    "mean_reproj_error": round(m.mean_reproj_error, 3) if np.isfinite(m.mean_reproj_error) else None,
                    "overlap_ssim": round(m.overlap_ssim, 4) if np.isfinite(m.overlap_ssim) else None,
                    "detect_s": round(m.detect_s, 3),
                    "match_s": round(m.match_s, 3),
                    "ransac_s": round(m.ransac_s, 3),
                    "pair_total_s": round(m.total_s, 3),
                    "stitch_total_s": round(result.stitch_s, 3),
                    "panorama_mean_inlier_ratio": round(result.mean_inlier_ratio, 4),
                }
            )
    df = save_table(RESULTS_DIR / "detector_comparison.csv", rows)
    _bar_comparison(df)
    return df


def _bar_comparison(df: pd.DataFrame) -> None:
    summary = df.groupby("detector", as_index=False).agg(
        inlier_ratio=("inlier_ratio", "mean"),
        n_inliers=("n_inliers", "mean"),
        pair_total_s=("pair_total_s", "mean"),
        overlap_ssim=("overlap_ssim", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    metrics = [
        ("inlier_ratio", "Mean inlier ratio"),
        ("pair_total_s", "Mean pair time (s)"),
        ("overlap_ssim", "Mean overlap SSIM"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(summary["detector"], summary[col], color=["#1f4e79", "#c47b17"])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "detector_comparison.png", dpi=140)
    plt.close(fig)


def _pair_from_main() -> tuple[np.ndarray, np.ndarray]:
    images = load_folder("scene_main")
    mid = len(images) // 2
    return images[mid - 1], images[mid]


def run_robustness() -> None:
    img_a, img_b = _pair_from_main()
    rotation_rows = []
    for angle in (0, 15, 30, 45, 90):
        transformed = img_b if angle == 0 else rotate(img_b, angle)
        for detector in ("SIFT", "ORB"):
            m = match_pair(img_a, transformed, detector=detector).metrics
            rotation_rows.append(
                {
                    "condition": "rotation",
                    "value": angle,
                    "unit": "deg",
                    "detector": detector,
                    **{k: getattr(m, k) for k in ("n_initial_matches", "n_inliers", "inlier_ratio", "mean_reproj_error", "total_s")},
                }
            )
    save_table(RESULTS_DIR / "robustness_rotation.csv", rotation_rows)
    _line_plot(rotation_rows, "value", "Rotation (degrees)", "robustness_rotation.png")

    scale_rows = []
    for factor in (0.5, 0.75, 1.0, 1.5):
        transformed = img_b if factor == 1.0 else scale(img_b, factor)
        for detector in ("SIFT", "ORB"):
            m = match_pair(img_a, transformed, detector=detector).metrics
            scale_rows.append(
                {
                    "condition": "scale",
                    "value": factor,
                    "unit": "x",
                    "detector": detector,
                    **{k: getattr(m, k) for k in ("n_initial_matches", "n_inliers", "inlier_ratio", "mean_reproj_error", "total_s")},
                }
            )
    save_table(RESULTS_DIR / "robustness_scale.csv", scale_rows)
    _line_plot(scale_rows, "value", "Scale factor", "robustness_scale.png")

    light_rows = []
    for gain, bias, label in ((1.0, 0, 1.0), (0.55, -20, 0.55), (0.35, -40, 0.35), (1.45, 25, 1.45)):
        transformed = img_b if gain == 1.0 and bias == 0 else illuminate(img_b, gain, bias)
        for detector in ("SIFT", "ORB"):
            m = match_pair(img_a, transformed, detector=detector).metrics
            light_rows.append(
                {
                    "condition": "illumination",
                    "value": label,
                    "unit": "gain",
                    "detector": detector,
                    **{k: getattr(m, k) for k in ("n_initial_matches", "n_inliers", "inlier_ratio", "mean_reproj_error", "total_s")},
                }
            )
    # Real lighting folder
    lighting = load_folder("scene_lighting")
    if len(lighting) >= 2:
        for detector in ("SIFT", "ORB"):
            m = match_pair(lighting[0], lighting[1], detector=detector).metrics
            light_rows.append(
                {
                    "condition": "illumination_photos",
                    "value": 0.72,
                    "unit": "scene_lighting",
                    "detector": detector,
                    **{k: getattr(m, k) for k in ("n_initial_matches", "n_inliers", "inlier_ratio", "mean_reproj_error", "total_s")},
                }
            )
    save_table(RESULTS_DIR / "robustness_illumination.csv", light_rows)
    synth = [r for r in light_rows if r["condition"] == "illumination"]
    _line_plot(synth, "value", "Brightness gain", "robustness_illumination.png")

    view_rows = []
    views = load_folder("scene_viewpoint")
    labels = ["front-oblique", "front-side"]
    for j, other in enumerate(views[1:]):
        for detector in ("SIFT", "ORB"):
            m = match_pair(views[0], other, detector=detector).metrics
            view_rows.append(
                {
                    "condition": "viewpoint",
                    "pair": labels[j] if j < len(labels) else f"0-{j+1}",
                    "detector": detector,
                    **{k: getattr(m, k) for k in ("n_initial_matches", "n_inliers", "inlier_ratio", "mean_reproj_error", "total_s", "overlap_ssim")},
                }
            )
    save_table(RESULTS_DIR / "robustness_viewpoint.csv", view_rows)
    _view_bar(view_rows)


def _line_plot(rows: list[dict], x_key: str, xlabel: str, filename: str) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for detector, part in df.groupby("detector"):
        part = part.sort_values(x_key)
        ax.plot(part[x_key], part["inlier_ratio"], marker="o", label=detector)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Inlier ratio")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=140)
    plt.close(fig)


def _view_bar(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    pairs = list(df["pair"].unique())
    x = np.arange(len(pairs))
    width = 0.35
    for i, detector in enumerate(("SIFT", "ORB")):
        vals = [df[(df.pair == p) & (df.detector == detector)]["inlier_ratio"].mean() for p in pairs]
        ax.bar(x + (i - 0.5) * width, vals, width, label=detector)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Inlier ratio")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_viewpoint.png", dpi=140)
    plt.close(fig)


def save_input_contact_sheet() -> None:
    images = load_folder("scene_main")
    rgb = [bgr_to_rgb(im) for im in images]
    n = len(rgb)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, im in zip(axes, rgb):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "input_scene_main.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Saving input contact sheet...")
    save_input_contact_sheet()
    print("Comparing SIFT vs ORB...")
    df = run_detector_comparison()
    print(df.to_string(index=False))
    print("Running robustness experiments...")
    run_robustness()
    print(f"Wrote figures to {FIGURES_DIR}")
    print(f"Wrote tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

"""SIFT and ORB keypoint detection and description."""

from __future__ import annotations

import cv2
import numpy as np

DETECTORS = ("SIFT", "ORB")


def create_detector(name: str, n_features: int = 2000):
    key = name.strip().upper()
    if key == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("cv2.SIFT_create is unavailable. Install opencv-python >= 4.4.")
        return cv2.SIFT_create(nfeatures=n_features)
    if key == "ORB":
        return cv2.ORB_create(nfeatures=n_features)
    raise ValueError(f"Unknown detector {name!r}. Choose SIFT or ORB.")


def detect_and_describe(
    gray: np.ndarray,
    detector_name: str = "SIFT",
    n_features: int = 2000,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    detector = create_detector(detector_name, n_features=n_features)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return list(keypoints or []), descriptors

"""Descriptor matching with Lowe's ratio test."""

from __future__ import annotations

import cv2
import numpy as np


def matcher_for(detector_name: str) -> cv2.BFMatcher:
    if detector_name.strip().upper() == "ORB":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def ratio_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    detector_name: str,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        return []
    matcher = matcher_for(detector_name)
    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return good


def match_point_arrays(
    keypoints_a: list[cv2.KeyPoint],
    keypoints_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    if not matches:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty
    pts_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches])
    return pts_a, pts_b

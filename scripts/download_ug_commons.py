#!/usr/bin/env python3
"""Download CC-licensed University of Ghana campus photos from Wikimedia Commons.

Images are resized (long edge 1600 px) and written into data/scenes/.
See data/ATTRIBUTION.md for authors and licences.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "data" / "scenes"
LONG_EDGE = 1600
UA = "FeatureBasedPanorama/1.0 (academic; https://github.com/qbentil/feature-based-panorama)"

# dest_folder, dest_name, Wikimedia file title
CATALOG: list[tuple[str, str, str]] = [
    ("scene_main", "01.jpg", "Balme Library 1.jpg"),
    ("scene_main", "02.jpg", "Balme Library 2.jpg"),
    ("scene_main", "03.jpg", "Balme Library 3.jpg"),
    ("scene_viewpoint", "01_front.jpg", "University of Ghana Dance Department 1.jpg"),
    ("scene_viewpoint", "02_oblique.jpg", "University of Ghana Dance Department 2.jpg"),
    ("scene_viewpoint", "03_side.jpg", "University of Ghana Dance Department 3.jpg"),
    ("scene_lighting", "01_daylight.jpg", "University of Ghana Night Market.jpg"),
    ("scene_lighting", "02_shade.jpg", "University of Ghana Night Market 2.jpg"),
    ("scene_lighting", "03_dim.jpg", "University of Ghana Night Market 3.jpg"),
    ("scene_second", "01.jpg", "Entrance to Commonwealth Hall Legon 1.jpg"),
    ("scene_second", "02.jpg", "Entrance to Commonwealth Hall Legon 2.jpg"),
    ("scene_second", "03.jpg", "Entrance to Commonwealth Hall Legon 3.jpg"),
    ("scene_second", "04.jpg", "Entrance to Commonwealth Hall Legon 4.jpg"),
    ("scene_second", "05.jpg", "Entrance to Commonwealth Hall Legon 5.jpg"),
]


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def commons_thumb(title: str, width: int) -> bytes:
    path = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{path}?width={width}"
    return fetch_bytes(url)


def resize_long_edge(bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= long_edge:
        return bgr
    scale = long_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("OpenCV could not decode a Commons download.")
    return img


def main() -> None:
    for folder, name, title in CATALOG:
        dest_dir = SCENES / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        print(f"Fetching {title} -> {dest.relative_to(ROOT)}")
        raw = commons_thumb(title, LONG_EDGE)
        bgr = resize_long_edge(decode_image(raw), LONG_EDGE)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Failed to encode {dest}")
        dest.write_bytes(encoded.tobytes())
    print(json.dumps({"wrote": len(CATALOG), "long_edge": LONG_EDGE}, indent=2))


if __name__ == "__main__":
    main()

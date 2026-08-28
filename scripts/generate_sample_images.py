#!/usr/bin/env python3
"""Generate overlapping sample scenes for panorama development.

These are synthetic stand-ins so the pipeline can run before real phone photos
are added. They include distinctive textured landmarks so SIFT/ORB can match.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "data" / "scenes"
TMP = ROOT / "data" / "raw"


# Tiny 5x7 glyphs so buildings have unique readable labels (no PIL needed).
_FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
}


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
    tmp = TMP / f"{path.stem}.ppm"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb8.shape
    with tmp.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb8.tobytes())
    subprocess.run(
        [
            "sips",
            "-s",
            "format",
            "jpeg",
            "-s",
            "formatOptions",
            str(quality),
            str(tmp),
            "--out",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    tmp.unlink(missing_ok=True)


def draw_text(img: np.ndarray, x: int, y: int, text: str, color, scale: int = 4) -> None:
    h, w, _ = img.shape
    color = np.asarray(color, dtype=np.uint8)
    cursor = x
    for ch in text.upper():
        glyph = _FONT.get(ch, _FONT[" "])
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                y0, x0 = y + r * scale, cursor + c * scale
                img[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def noise(shape, rng: np.random.Generator, amp: float) -> np.ndarray:
    return rng.normal(0, amp, size=shape)


def fill_rect(img, x0, y0, x1, y1, color) -> None:
    h, w, _ = img.shape
    xa, xb = max(0, int(x0)), min(w, int(x1))
    ya, yb = max(0, int(y0)), min(h, int(y1))
    if xb > xa and yb > ya:
        img[ya:yb, xa:xb] = color


def campus_mural(rng: np.random.Generator) -> np.ndarray:
    """Wide outdoor courtyard: buildings, trees, banners, pavement."""
    w, h = 4200, 1000
    img = np.zeros((h, w, 3), dtype=np.float32)

    yy = np.linspace(0, 1, h)[:, None]
    sky = np.dstack(
        [
            70 + 90 * (1 - yy) + np.zeros((h, w)),
            130 + 70 * (1 - yy) + np.zeros((h, w)),
            190 + 50 * (1 - yy) + np.zeros((h, w)),
        ]
    )
    img[:] = sky

    # Pavement
    fill_rect(img, 0, 720, w, h, (92, 90, 86))
    img[720:, :] += noise((h - 720, w, 3), rng, 8)

    # Distant hills
    xs = np.arange(w)
    hill = 430 + 40 * np.sin(xs / 280) + 25 * np.sin(xs / 110)
    for x in range(w):
        y0 = int(hill[x])
        img[y0:720, x] = (58, 110, 62)
    img[430:720, :] += noise((290, w, 3), rng, 6)

    buildings = [
        (80, 260, 520, 720, (168, 92, 72), "HALL A"),
        (560, 180, 1180, 720, (210, 196, 178), "LIBRARY"),
        (1240, 300, 1680, 720, (72, 118, 148), "LAB 2"),
        (1760, 140, 2480, 720, (186, 142, 96), "FACULTY"),
        (2560, 240, 3180, 720, (120, 88, 128), "CHAPEL"),
        (3260, 200, 3920, 720, (96, 140, 108), "CAFE"),
    ]
    for x0, y0, x1, y1, color, label in buildings:
        body = np.array(color, dtype=np.float32)
        fill_rect(img, x0, y0, x1, y1, body)
        region = img[y0:y1, x0:x1]
        region += noise(region.shape, rng, 7)
        # brick rows
        for by in range(y0 + 18, y1 - 10, 22):
            img[by : by + 2, x0:x1] = np.clip(body * 0.78, 0, 255)
        # windows
        win_w, win_h = 28, 38
        gap_x, gap_y = 48, 58
        for wy in range(y0 + 50, y1 - 90, gap_y):
            for wx in range(x0 + 24, x1 - 30, gap_x):
                shade = 40 + int(rng.integers(0, 50))
                fill_rect(img, wx, wy, wx + win_w, wy + win_h, (shade + 20, shade + 35, shade + 55))
                fill_rect(img, wx + 2, wy + 2, wx + win_w - 2, wy + win_h // 2, (shade + 70, shade + 90, shade + 120))
        # roof
        fill_rect(img, x0 - 12, y0 - 28, x1 + 12, y0 + 6, (48, 42, 40))
        # banner
        fill_rect(img, x0 + 40, y0 + 16, x1 - 40, y0 + 70, (18, 28, 48))
        draw_text(img, x0 + 56, y0 + 26, label, (245, 230, 90), scale=5)

    # Trees
    for cx, cy, r, green in [
        (500, 700, 70, (28, 110, 46)),
        (1210, 690, 55, (22, 96, 40)),
        (2500, 705, 80, (34, 124, 52)),
        (3210, 688, 60, (26, 108, 44)),
        (4000, 700, 75, (30, 116, 50)),
    ]:
        yy, xx = np.ogrid[-r:r, -r:r]
        mask = xx * xx + yy * yy <= r * r
        y0, x0 = cy - r, cx - r
        patch = img[y0 : y0 + 2 * r, x0 : x0 + 2 * r]
        foliage = np.array(green, dtype=np.float32) + noise((2 * r, 2 * r, 3), rng, 14)
        patch[mask] = foliage[mask]
        fill_rect(img, cx - 8, cy, cx + 8, 780, (72, 48, 28))

    # Unique circular crests (good keypoints)
    for cx, cy, col in [(900, 520, (180, 30, 30)), (2100, 430, (20, 80, 160)), (2800, 500, (200, 160, 20)), (3600, 470, (160, 40, 120))]:
        rr = 42
        yy, xx = np.ogrid[-rr:rr, -rr:rr]
        mask = xx * xx + yy * yy <= rr * rr
        ring = (xx * xx + yy * yy <= rr * rr) & (xx * xx + yy * yy >= (rr - 8) ** 2)
        img[cy - rr : cy + rr, cx - rr : cx + rr][mask] = col
        img[cy - rr : cy + rr, cx - rr : cx + rr][ring] = (250, 250, 240)

    # Path lines
    for x in range(0, w, 70):
        fill_rect(img, x, 860, x + 36, 872, (210, 200, 80))

    return np.clip(img, 0, 255)


def indoor_mural(rng: np.random.Generator) -> np.ndarray:
    """Second scene: noticeboard / corridor posters (different texture)."""
    w, h = 3600, 900
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = (214, 198, 170)
    img += noise(img.shape, rng, 5)
    # wall panels
    for x in range(0, w, 240):
        fill_rect(img, x, 0, x + 8, h, (168, 150, 126))
    fill_rect(img, 0, 780, w, h, (96, 78, 62))
    img[780:] += noise((h - 780, w, 3), rng, 6)

    posters = [
        (120, 80, 620, 520, (40, 70, 130), "NOTICE"),
        (700, 120, 1180, 560, (150, 40, 40), "SPORTS"),
        (1280, 60, 1860, 540, (30, 110, 70), "GREEN"),
        (1960, 100, 2480, 580, (120, 80, 20), "THESIS"),
        (2580, 70, 3180, 530, (80, 40, 110), "CLUB"),
    ]
    for x0, y0, x1, y1, color, label in posters:
        fill_rect(img, x0 - 14, y0 - 14, x1 + 14, y1 + 14, (236, 228, 210))
        fill_rect(img, x0, y0, x1, y1, color)
        region = img[y0:y1, x0:x1]
        region += noise(region.shape, rng, 10)
        # inner photo-like blocks
        for i in range(4):
            px = x0 + 30 + (i % 2) * ((x1 - x0) // 2 - 20)
            py = y0 + 80 + (i // 2) * 160
            fill_rect(img, px, py, px + (x1 - x0) // 2 - 50, py + 130, tuple(int(c) for c in np.clip(np.array(color) * 0.55 + rng.integers(0, 40, 3), 0, 255)))
        fill_rect(img, x0 + 20, y1 - 70, x1 - 20, y1 - 18, (245, 240, 220))
        draw_text(img, x0 + 36, y1 - 62, label, (20, 20, 20), scale=4)

    # clock / exit signs as extra unique features
    for cx, cy, label, col in [(640, 640, "EXIT", (180, 20, 20)), (1900, 660, "INFO", (20, 80, 40)), (3000, 650, "OPEN", (20, 40, 140))]:
        fill_rect(img, cx, cy, cx + 220, cy + 70, col)
        draw_text(img, cx + 20, cy + 16, label, (250, 250, 250), scale=5)

    return np.clip(img, 0, 255)


def crop(img: np.ndarray, x: int, width: int) -> np.ndarray:
    x = max(0, min(x, img.shape[1] - width))
    return img[:, x : x + width].copy()


def sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    top = Ia * (1 - wx)[..., None] + Ib * wx[..., None]
    bot = Ic * (1 - wx)[..., None] + Id * wx[..., None]
    return top * (1 - wy)[..., None] + bot * wy[..., None]


def perspective_view(img: np.ndarray, x0: int, width: int, skew: float) -> np.ndarray:
    """Simulate a side viewpoint by sampling a trapezoid from the mural."""
    h, w, _ = img.shape
    out_h, out_w = h, width
    ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    # horizontal squeeze that grows with y (ground-plane-ish)
    t = ys / max(out_h - 1, 1)
    src_x = x0 + xs * (1.0 - skew * 0.25) + t * xs * skew + t * skew * 80
    src_y = ys * (1.0 - 0.04 * skew) + 20 * skew * (xs / out_w)
    return sample_bilinear(img, src_x, src_y)


def adjust_light(img: np.ndarray, gain: float, shadow_left: bool = False) -> np.ndarray:
    out = img.astype(np.float32) * gain
    if shadow_left:
        ramp = np.linspace(0.45, 1.0, img.shape[1])[None, :, None]
        out *= ramp
    return np.clip(out, 0, 255)


def main() -> None:
    rng = np.random.default_rng(608)
    campus = campus_mural(rng)
    indoor = indoor_mural(rng)

    view_w = 1400
    # ~50% overlap steps
    main_xs = [0, 700, 1400, 2100, 2800]
    for i, x in enumerate(main_xs, start=1):
        save_jpg(SCENES / "scene_main" / f"{i:02d}.jpg", crop(campus, x, view_w))

    # Wider baseline = stronger viewpoint change (same courtyard)
    save_jpg(SCENES / "scene_viewpoint" / "01_front.jpg", crop(campus, 1400, view_w))
    save_jpg(
        SCENES / "scene_viewpoint" / "02_oblique.jpg",
        perspective_view(campus, 1320, view_w, skew=0.28),
    )
    save_jpg(
        SCENES / "scene_viewpoint" / "03_side.jpg",
        perspective_view(campus, 1550, view_w, skew=0.42),
    )

    pair = crop(campus, 1400, view_w)
    save_jpg(SCENES / "scene_lighting" / "01_daylight.jpg", pair)
    save_jpg(SCENES / "scene_lighting" / "02_shade.jpg", adjust_light(pair, 0.72, shadow_left=True))
    save_jpg(SCENES / "scene_lighting" / "03_dim.jpg", adjust_light(pair, 0.42))

    indoor_w = 1200
    for i, x in enumerate([0, 600, 1200, 1800, 2400], start=1):
        save_jpg(SCENES / "scene_second" / f"{i:02d}.jpg", crop(indoor, x, indoor_w))

    print("Wrote sample scenes to", SCENES)


if __name__ == "__main__":
    main()

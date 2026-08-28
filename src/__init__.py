"""Classical feature-based panorama pipeline.

Heavy OpenCV imports live in src.pipeline and are loaded only when needed,
so pages such as the PDF report can import src.paths on Streamlit Cloud
without requiring cv2 at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["match_pair", "stitch_images"]


def __getattr__(name: str) -> Any:
    if name in {"match_pair", "stitch_images"}:
        from src.pipeline import match_pair, stitch_images

        return match_pair if name == "match_pair" else stitch_images
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

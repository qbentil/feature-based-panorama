#!/usr/bin/env python3
"""Compile report/main.tex with the CVPR LaTeX template via Tectonic."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
TEX = REPORT / "main.tex"
SUBMISSION_TEX = REPORT / "submission.tex"
COMPILED = REPORT / "main.pdf"
OUT_PDF = REPORT / "feature_based_panorama.pdf"
SUBMISSION_PDF = REPORT / "Bentil_feature_based_panorama_submission.pdf"


def tectonic_bin() -> str:
    exe = shutil.which("tectonic")
    if not exe:
        raise FileNotFoundError(
            "tectonic is required to compile the CVPR report "
            "(brew install tectonic)."
        )
    return exe


def main() -> None:
    if not TEX.exists():
        raise FileNotFoundError(TEX)
    tex = tectonic_bin()
    # Classic interface: no Tectonic.toml required; runs bibtex via ieee.bst.
    subprocess.run([tex, TEX.name], check=True, cwd=str(REPORT))
    if not COMPILED.exists() or COMPILED.stat().st_size == 0:
        raise RuntimeError("Tectonic did not produce main.pdf")
    shutil.copy2(COMPILED, OUT_PDF)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")

    if SUBMISSION_TEX.exists():
        subprocess.run([tex, SUBMISSION_TEX.name], check=True, cwd=str(REPORT))
        compiled_sub = REPORT / "submission.pdf"
        if compiled_sub.exists() and compiled_sub.stat().st_size > 0:
            shutil.copy2(compiled_sub, SUBMISSION_PDF)
            print(f"Wrote {SUBMISSION_PDF} ({SUBMISSION_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

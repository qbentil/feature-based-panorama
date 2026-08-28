# Feature-based image matching and automatic panorama construction

**Author:** Shadrack Bentil  
MSc Computer Science, University of Ghana, Legon  
[sbentil005@st.ug.edu.gh](mailto:sbentil005@st.ug.edu.gh) · [GitHub @qbentil](https://github.com/qbentil)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4+-green.svg)](https://opencv.org/)

Classical **feature detection → description → matching → RANSAC → homography → stitch**. Implemented with OpenCV primitives. The system does **not** call `cv2.Stitcher`, YOLO, or any pretrained recogniser.

Full write-up: [report/main.tex](report/main.tex) (CVPR LaTeX) · PDF: [report/feature_based_panorama.pdf](report/feature_based_panorama.pdf)

---

## Abstract

This project recovers a single panoramic image from three or more overlapping views of a scene. Distinctive keypoints are detected and described with SIFT and ORB, matched with Lowe's ratio test, and filtered with RANSAC to estimate a homography. Every image is warped into the middle view's coordinate system and the overlap is feather-blended. On three Balme Library photographs, both detectors produce a complete panorama (mean inlier ratio ~0.95 for SIFT and ~0.91 for ORB). Both methods fail under severe underexposure and when viewpoint change leaves the overlapping region.

---

## Method

1. Prepare each photo (resize, optional Gaussian blur, CLAHE).
2. Detect and describe landmarks with **SIFT** or **ORB**.
3. Match descriptors with kNN and Lowe's ratio test.
4. Estimate a homography with **RANSAC** and drop outliers.
5. Warp every photo into the **middle** photo's coordinate system.
6. Feather-blend the overlap into a panorama.

An interactive Streamlit app walks through each stage, compares the two detectors, and runs rotation, scale, illumination, and viewpoint sweeps.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiments.py
python scripts/build_report.py   # requires tectonic (brew install tectonic)
streamlit run streamlit_app.py
```

Open the app, pick **scene main**, click **Stitch**, then step through keypoints → raw matches → RANSAC inliers → warp → panorama. **Project report** embeds the PDF.

---

## Repository layout

| Path | Role |
|---|---|
| `src/` | Pipeline used by both the app and the experiment script |
| `app_pages/` | Streamlit UI |
| `scripts/run_experiments.py` | SIFT vs ORB table and robustness figures |
| `scripts/download_ug_commons.py` | Fetch CC campus photos into `data/scenes/` |
| `scripts/build_report.py` | Compile CVPR LaTeX (`report/main.tex`) with Tectonic |
| `data/scenes/` | Input photos (Wikimedia Commons, Legon) |
| `data/ATTRIBUTION.md` | Photo authors and licences |
| `report/main.tex` | Technical report (CVPR 10pt two-column template) |
| `report/feature_based_panorama.pdf` | Typeset PDF |
| `report/figures/` | Generated evidence |

---

## Dataset

Bundled scenes in `data/scenes/` are real University of Ghana campus photographs from Wikimedia Commons (CC BY / CC BY-SA). Attribution is in [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md). Capture notes are in [`data/README.md`](data/README.md).

Experiments record, for each detector:

- number of keypoints
- number of initial matches
- number of RANSAC inliers
- inlier ratio
- processing time
- panorama quality (mean reprojection error and overlap SSIM)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{bentil2026panorama,
  title        = {Feature-based Image Matching and Automatic Panorama Construction},
  author       = {Bentil, Shadrack},
  year         = {2026},
  institution  = {University of Ghana, Legon},
  note         = {Department of Computer Science},
  url          = {https://github.com/qbentil}
}
```

---

## Contact

**Shadrack Bentil**  
MSc Computer Science  
University of Ghana, Legon

- Email: [sbentil005@st.ug.edu.gh](mailto:sbentil005@st.ug.edu.gh)
- GitHub: [@qbentil](https://github.com/qbentil)
- LinkedIn: [linkedin.com/in/bentil](https://linkedin.com/in/bentil)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## References

1. D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, 2004.
2. E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2011, pp. 2564–2571.
3. M. A. Fischler and R. C. Bolles, "Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography," *Communications of the ACM*, vol. 24, no. 6, pp. 381–395, 1981.
4. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2003.
5. M. Brown and D. G. Lowe, "Automatic panoramic image stitching using invariant features," *International Journal of Computer Vision*, vol. 74, no. 1, pp. 59–73, 2007.
6. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

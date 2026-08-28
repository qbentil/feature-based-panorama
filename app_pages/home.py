import streamlit as st

st.title("How a panorama is built")
st.caption("Classical computer vision — not a pretrained recogniser")

st.markdown(
    """
Phones make panoramas look like magic. Under the hood they are doing a very
ordinary thing: **find the same landmarks in two photos, throw away the bad
guesses, then stretch one photo until those landmarks line up.**

This app walks through that recipe with OpenCV. Nothing here is YOLO, nothing
here is a black-box `Stitcher` class. Each step is a classical technique.
"""
)

cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.markdown("**1. Landmarks**")
        st.caption(
            "SIFT and ORB look for corners and blobs that stay recognisable when "
            "you move the camera. Think of unique bricks, window corners, signs."
        )
with cols[1]:
    with st.container(border=True):
        st.markdown("**2. Honest matches**")
        st.caption(
            "Many landmark pairs are wrong. RANSAC keeps only the matches that "
            "agree on one geometric stretch — the homography."
        )
with cols[2]:
    with st.container(border=True):
        st.markdown("**3. One canvas**")
        st.caption(
            "Each photo is warped onto the middle photo's coordinate system and "
            "the overlap is blended so the join is hard to see."
        )

st.markdown(
    """
```mermaid
flowchart LR
  photos[Overlapping photos] --> prep[Prepare]
  prep --> detect[Detect and describe]
  detect --> match[Match]
  match --> ransac[RANSAC]
  ransac --> H[Homography]
  H --> warp[Warp to the middle photo]
  warp --> blend[Feather blend]
  blend --> pano[Panorama]
```
"""
)

st.subheader("What you can do here")
st.markdown(
    """
- **Build a panorama** — drop in three or more overlapping photos and step through keypoints, raw matches, RANSAC inliers, the warp, and the final stitch.
- **Compare detectors** — SIFT versus ORB on the same pair: keypoints, matches, inliers, inlier ratio, time, panorama quality.
- **Robustness lab** — rotate, scale, or darken one photo and watch the inlier ratio fall. Viewpoint uses a separate photo set.
- **Saved results** — tables and figures written by `scripts/run_experiments.py`.
- **Project report** — the full technical report as a PDF.

Live app: [feature-based-panorama.streamlit.app](https://feature-based-panorama.streamlit.app/) · source: [github.com/qbentil/feature-based-panorama](https://github.com/qbentil/feature-based-panorama)
"""
)

st.info(
    "The bundled scenes are real University of Ghana campus photographs from "
    "Wikimedia Commons (Balme Library, Dance Department, Night Market, "
    "Commonwealth Hall). They were not all shot as a dedicated panorama, so "
    "some sets overlap much more than others. Drop your own overlapping photos "
    "in the same folders when you can — the method does not change.",
    icon=":material/info:",
)

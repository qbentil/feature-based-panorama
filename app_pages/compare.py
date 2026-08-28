import pandas as pd
import streamlit as st

from app_lib import cached_stitch, encode_bgr, load_images_from_dir, scene_options, show_rgb
from src.paths import RESULTS_DIR

st.title("Compare SIFT and ORB")
st.caption("Same photos, two detectors. Keypoints, matches, inliers, time, and panorama quality.")

scenes = scene_options()
names = list(scenes.keys())
default_idx = names.index("scene main") if "scene main" in names else 0
chosen = st.selectbox("Scene", names, index=default_idx)

with st.form("compare_form"):
    max_side = st.select_slider("Long-edge size", options=[800, 1000, 1200], value=1000)
    run = st.form_submit_button("Run both detectors", type="primary", icon=":material/compare_arrows:")

if run:
    images = load_images_from_dir(scenes[chosen])
    if len(images) < 2:
        st.error("This folder needs at least two images.")
        st.stop()
    payload = tuple(encode_bgr(img) for img in images)
    rows = []
    panoramas = {}
    with st.spinner("Running SIFT, then ORB…"):
        for detector in ("SIFT", "ORB"):
            result = cached_stitch(payload, detector, 0.75, 5.0, int(max_side), 3, True)
            panoramas[detector] = result.panorama_rgb
            for idx, pair in enumerate(result.pair_results):
                m = pair.metrics
                rows.append(
                    {
                        "Detector": detector,
                        "Pair": f"{idx + 1}–{idx + 2}",
                        "Keypoints left": m.n_keypoints_a,
                        "Keypoints right": m.n_keypoints_b,
                        "Initial matches": m.n_initial_matches,
                        "RANSAC inliers": m.n_inliers,
                        "Inlier ratio": m.inlier_ratio,
                        "Reprojection error (px)": m.mean_reproj_error,
                        "Overlap SSIM": m.overlap_ssim,
                        "Pair time (s)": m.total_s,
                        "Stitch time (s)": result.stitch_s,
                    }
                )
    st.session_state.compare_rows = rows
    st.session_state.compare_panoramas = panoramas

rows = st.session_state.get("compare_rows")
if not rows:
    saved = RESULTS_DIR / "detector_comparison.csv"
    if saved.exists():
        st.info("No live run yet. Showing the last `run_experiments.py` table.")
        st.dataframe(pd.read_csv(saved), hide_index=True)
    else:
        st.info("Pick a scene and run both detectors.")
    st.stop()

df = pd.DataFrame(rows)
st.dataframe(
    df,
    hide_index=True,
    column_config={
        "Inlier ratio": st.column_config.NumberColumn(format="%.2f"),
        "Reprojection error (px)": st.column_config.NumberColumn(format="%.2f"),
        "Overlap SSIM": st.column_config.NumberColumn(format="%.3f"),
        "Pair time (s)": st.column_config.NumberColumn(format="%.2f"),
        "Stitch time (s)": st.column_config.NumberColumn(format="%.2f"),
    },
)

summary = (
    df.groupby("Detector", as_index=False)
    .agg(
        **{
            "Mean inlier ratio": ("Inlier ratio", "mean"),
            "Mean pair time (s)": ("Pair time (s)", "mean"),
            "Mean overlap SSIM": ("Overlap SSIM", "mean"),
        }
    )
)
st.bar_chart(summary, x="Detector", y="Mean inlier ratio")

panoramas = st.session_state.get("compare_panoramas") or {}
if panoramas:
    left, right = st.columns(2)
    if "SIFT" in panoramas:
        with left:
            st.markdown("**SIFT panorama**")
            show_rgb(panoramas["SIFT"])
    if "ORB" in panoramas:
        with right:
            st.markdown("**ORB panorama**")
            show_rgb(panoramas["ORB"])

st.markdown(
    """
SIFT usually keeps more inliers under rotation and scale because its descriptor
is built for that. ORB is a binary descriptor: it is faster, and it often
drops first when the lighting changes.
"""
)

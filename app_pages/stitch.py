import numpy as np
import streamlit as st

from app_lib import (
    cached_stitch,
    decode_uploads,
    encode_bgr,
    load_images_from_dir,
    metric_row,
    scene_options,
    show_rgb,
    show_thumbs,
    square_thumb,
)

st.title("Build a panorama")
st.caption("Upload overlapping photos or pick a bundled scene, then walk the pipeline one step at a time.")

scenes = scene_options()
source = st.segmented_control(
    "Image source",
    options=["Bundled scene", "Upload photos"],
    default="Bundled scene",
    required=True,
)

images: list[np.ndarray] = []
if source == "Upload photos":
    files = st.file_uploader(
        "Overlapping photos (at least 3, left to right)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if files:
        files = sorted(files, key=lambda f: f.name)
        images = decode_uploads(files)
else:
    names = list(scenes.keys())
    default_idx = names.index("scene main") if "scene main" in names else 0
    chosen = st.selectbox("Scene", names, index=default_idx)
    images = load_images_from_dir(scenes[chosen])

if images:
    show_thumbs(images)

with st.form("stitch_form"):
    c1, c2, c3 = st.columns(3)
    detector = c1.segmented_control("Detector", options=["SIFT", "ORB"], default="SIFT", required=True)
    ratio = c2.slider("Lowe ratio", 0.5, 0.9, 0.75, 0.05)
    ransac = c3.slider("RANSAC threshold (px)", 1.0, 10.0, 5.0, 0.5)
    c4, c5, c6 = st.columns(3)
    max_side = c4.select_slider("Long-edge size", options=[800, 1000, 1200, 1600], value=1000)
    blur = c5.select_slider("Gaussian blur", options=[0, 3, 5], value=3)
    use_clahe = c6.toggle("CLAHE lighting prep", value=True)
    submitted = st.form_submit_button("Stitch", type="primary", icon=":material/play_arrow:")

if submitted:
    if len(images) < 2:
        st.error("Need at least two overlapping images.")
        st.stop()
    if detector is None:
        st.error("Pick a detector.")
        st.stop()
    with st.spinner("Detecting landmarks, rejecting bad matches, warping…"):
        payload = tuple(encode_bgr(img) for img in images)
        try:
            result = cached_stitch(
                payload,
                detector,
                float(ratio),
                float(ransac),
                int(max_side),
                int(blur),
                bool(use_clahe),
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    st.session_state.stitch_result = result

result = st.session_state.get("stitch_result")
if result is None:
    st.info("Choose images and click Stitch.", icon=":material/panorama_horizontal:")
    st.stop()

st.subheader("Match metrics")
m0 = result.pair_results[0].metrics
metric_row(
    m0,
    extra={
        "Mean inlier ratio across pairs": f"{result.mean_inlier_ratio:.2%}",
        "Stitch time": f"{result.stitch_s:.2f} s",
        "Reference image": f"#{result.ref_index + 1} (middle)",
    },
)

steps = [
    "Input photos",
    "Keypoints",
    "Raw matches",
    "RANSAC inliers",
    "Warped overlay",
    "Panorama",
]
step = st.segmented_control("Show step", options=steps, default="Panorama", required=True)
pair_labels = [f"Pair {i + 1}–{i + 2}" for i in range(len(result.pair_results))]
pair_i = 0
if step != "Panorama" and step != "Input photos":
    pair_i = st.selectbox(
        "Which consecutive pair",
        range(len(pair_labels)),
        format_func=lambda i: pair_labels[i],
    )
pair = result.pair_results[pair_i]

if step == "Input photos":
    st.markdown("These are neighbouring views of the same scene. The overlap is the puzzle-piece region.")
    n = min(5, len(result.colours))
    cols = st.columns(n)
    for col, img in zip(cols, result.colours[:n]):
        col.image(square_thumb(img), caption=f"{img.shape[1]}×{img.shape[0]}")
elif step == "Keypoints":
    st.markdown("Each circle is a landmark the detector thinks it could recognise again in the next photo.")
    c1, c2 = st.columns(2)
    with c1:
        show_rgb(pair.keypoints_a_vis, f"{pair.metrics.n_keypoints_a} keypoints")
    with c2:
        show_rgb(pair.keypoints_b_vis, f"{pair.metrics.n_keypoints_b} keypoints")
elif step == "Raw matches":
    st.markdown("Green lines are later confirmed by RANSAC; red lines are the guesses it will throw away.")
    show_rgb(pair.matches_raw_vis, "Before RANSAC")
    metric_row(pair.metrics)
elif step == "RANSAC inliers":
    st.markdown("Only matches that agree on one homography survive. That is the geometry we trust.")
    show_rgb(pair.matches_inlier_vis, "After RANSAC")
    metric_row(pair.metrics)
elif step == "Warped overlay":
    st.markdown("The left photo is stretched into the right photo's frame. If the landmarks were right, edges line up.")
    if pair.overlay_vis is None:
        st.warning("Homography was not estimated for this pair.")
    else:
        show_rgb(pair.overlay_vis, "Aligned overlay")
else:
    st.markdown(
        "Every photo is warped onto the **middle** photo, then the overlap is "
        "feather-blended. Seams should be hard to see if overlap was generous."
    )
    show_rgb(result.panorama_rgb, f"{result.detector} panorama")
    st.caption(
        f"Mean inlier ratio {result.mean_inlier_ratio:.2%} · "
        f"mean reprojection error {result.mean_reproj_error:.2f} px · "
        f"overlap SSIM {result.overlap_ssim:.3f}"
        if np.isfinite(result.mean_reproj_error)
        else f"Mean inlier ratio {result.mean_inlier_ratio:.2%}"
    )

import pandas as pd
import streamlit as st

from app_lib import cached_pair, encode_bgr, load_images_from_dir, scene_options, show_rgb, transform_bgr
from src.paths import DATA_SCENES, RESULTS_DIR

st.title("Robustness lab")
st.caption("Change one photo and watch how many landmark pairs RANSAC still believes.")

mode = st.segmented_control(
    "What to stress",
    options=["Rotation", "Scale", "Illumination", "Viewpoint"],
    default="Rotation",
    required=True,
)

scenes = scene_options()
main_key = "scene main" if "scene main" in scenes else next(iter(scenes))
images = load_images_from_dir(scenes[main_key])
if len(images) < 2:
    st.error("Need a bundled scene with at least two images.")
    st.stop()
mid = len(images) // 2
img_a, img_b = images[mid - 1], images[mid]

if mode == "Viewpoint":
    view_dir = DATA_SCENES / "scene_viewpoint"
    views = load_images_from_dir(view_dir)
    if len(views) < 2:
        st.warning("Put viewpoint photos in data/scenes/scene_viewpoint/.")
        st.stop()
    st.markdown("Front view versus a side step. This is the case we cannot fake with a simple rotate.")
    detector = st.segmented_control("Detector", options=["SIFT", "ORB"], default="SIFT", required=True)
    if st.button("Match viewpoint pair", type="primary"):
        if detector is None:
            st.stop()
        with st.spinner("Matching…"):
            pair = cached_pair(
                encode_bgr(views[0]),
                encode_bgr(views[-1]),
                detector,
                0.75,
                5.0,
                1000,
                3,
                True,
            )
        st.metric("Inlier ratio", f"{pair.metrics.inlier_ratio:.2%}")
        show_rgb(pair.matches_inlier_vis, "RANSAC inliers")
        saved = RESULTS_DIR / "robustness_viewpoint.csv"
        if saved.exists():
            st.dataframe(pd.read_csv(saved), hide_index=True)
    elif (RESULTS_DIR / "robustness_viewpoint.csv").exists():
        st.dataframe(pd.read_csv(RESULTS_DIR / "robustness_viewpoint.csv"), hide_index=True)
        st.image(str(RESULTS_DIR.parent / "figures" / "robustness_viewpoint.png"))
    st.stop()

st.markdown("The left photo stays still. The right photo is transformed, then we match again.")
with st.form("robust_form"):
    detector = st.segmented_control("Detector", options=["SIFT", "ORB"], default="SIFT", required=True)
    if mode == "Rotation":
        amounts = [0, 15, 30, 45, 90]
        unit = "°"
        kind = "rotation"
    elif mode == "Scale":
        amounts = [0.5, 0.75, 1.0, 1.5]
        unit = "×"
        kind = "scale"
    else:
        amounts = [0.35, 0.55, 1.0, 1.45]
        unit = " gain"
        kind = "illumination"
    run = st.form_submit_button(f"Sweep {mode.lower()}", type="primary", icon=":material/science:")

if run:
    if detector is None:
        st.stop()
    rows = []
    a_bytes = encode_bgr(img_a)
    with st.spinner(f"Matching across {mode.lower()}…"):
        for amount in amounts:
            transformed = transform_bgr(img_b, kind, float(amount))
            pair = cached_pair(
                a_bytes,
                encode_bgr(transformed),
                detector,
                0.75,
                5.0,
                1000,
                3,
                True,
            )
            rows.append(
                {
                    mode: amount,
                    "Initial matches": pair.metrics.n_initial_matches,
                    "Inliers": pair.metrics.n_inliers,
                    "Inlier ratio": pair.metrics.inlier_ratio,
                    "Time (s)": pair.metrics.total_s,
                }
            )
            if amount == amounts[len(amounts) // 2]:
                show_rgb(pair.matches_inlier_vis, f"Inliers at {amount}{unit}")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    st.line_chart(df, x=mode, y="Inlier ratio")
else:
    mapping = {
        "Rotation": "robustness_rotation.csv",
        "Scale": "robustness_scale.csv",
        "Illumination": "robustness_illumination.csv",
    }
    path = RESULTS_DIR / mapping[mode]
    fig = RESULTS_DIR.parent / "figures" / mapping[mode].replace(".csv", ".png")
    if path.exists():
        st.caption("Last experiment sweep from `scripts/run_experiments.py`.")
        st.dataframe(pd.read_csv(path), hide_index=True)
        if fig.exists():
            st.image(str(fig))
    else:
        st.info("Run a sweep, or generate saved curves with `python scripts/run_experiments.py`.")

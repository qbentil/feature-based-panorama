from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Feature-based panorama construction",
    page_icon=":material/panorama:",
    layout="wide",
)

st.session_state.setdefault("stitch_result", None)
st.session_state.setdefault("compare_rows", None)

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="How it works", icon=":material/home:"),
        st.Page("app_pages/stitch.py", title="Build a panorama", icon=":material/panorama_horizontal:"),
        st.Page("app_pages/compare.py", title="Compare detectors", icon=":material/compare_arrows:"),
        st.Page("app_pages/robustness.py", title="Robustness lab", icon=":material/science:"),
        st.Page("app_pages/results.py", title="Saved results", icon=":material/photo_library:"),
        st.Page("app_pages/report.py", title="Project report", icon=":material/description:"),
    ],
    position="top",
)
page.run()

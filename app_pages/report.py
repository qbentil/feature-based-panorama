import streamlit as st

from src.paths import REPORT_DIR

PDF_PATH = REPORT_DIR / "feature_based_panorama.pdf"

st.title("Project report")
st.caption("Shadrack Bentil · University of Ghana, Legon · CVPR LaTeX template")

if not PDF_PATH.exists():
    st.warning(
        "The PDF has not been built yet. From the project folder run "
        "`python scripts/build_report.py`."
    )
    st.stop()

data = PDF_PATH.read_bytes()
st.download_button(
    "Download PDF",
    data=data,
    file_name=PDF_PATH.name,
    mime="application/pdf",
    icon=":material/download:",
)

try:
    st.pdf(str(PDF_PATH), height=800)
except Exception:
    st.info("Install the PDF viewer with `pip install streamlit-pdf`, or use the download button above.")
    st.download_button(
        "Download PDF (fallback)",
        data=data,
        file_name=PDF_PATH.name,
        mime="application/pdf",
        key="pdf_fallback",
        icon=":material/download:",
    )

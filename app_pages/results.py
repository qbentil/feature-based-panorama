import pandas as pd
import streamlit as st

from src.paths import FIGURES_DIR, RESULTS_DIR

st.title("Saved results")
st.caption("Figures and tables written by `scripts/run_experiments.py` for the report.")

if not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob("*.csv")):
    st.warning(
        "No experiment outputs yet. From the project folder run: "
        "`python scripts/run_experiments.py`"
    )
    st.stop()

csvs = sorted(RESULTS_DIR.glob("*.csv"))
chosen = st.selectbox("Table", [p.name for p in csvs])
st.dataframe(pd.read_csv(RESULTS_DIR / chosen), hide_index=True)

st.subheader("Figures")
figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
if not figures:
    st.info("No figures found.")
else:
    labels = [p.stem.replace("_", " ") for p in figures]
    pick = st.selectbox("Figure", labels)
    path = figures[labels.index(pick)]
    st.image(str(path), caption=path.name)

    st.markdown("**Gallery**")
    for i in range(0, len(figures), 3):
        cols = st.columns(3)
        for col, fig in zip(cols, figures[i : i + 3]):
            col.image(str(fig), caption=fig.stem.replace("_", " "))

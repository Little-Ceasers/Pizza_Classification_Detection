import streamlit as st
from pages import Live_Detection, Best_Capture

st.set_page_config(page_title="Pizza QC System", layout="wide")

PAGES = {
"Live Monitoring": Live_Detection,
"Quality Analysis": Best_Capture
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
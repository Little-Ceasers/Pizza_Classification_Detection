import streamlit as st
from pages import Live_Detection, Best_Capture
import os



st.set_page_config(page_title="Pizza QC System", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Detection", "Best Capture"])

if not os.getenv("EBay_CAMERA_IP"):
    st.warning("Camera IP not set in environment!")

if page == "Live Detection":
    Live_Detection.main()
elif page == "Best Capture":
    Best_Capture.main()
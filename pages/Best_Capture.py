# pages/Best_Capture.py

import streamlit as st
import os
from modules.utils import classify_pizza, generate_gemini_analysis

st.set_page_config(page_title="Pizza QC System", layout="wide")

def main():
    st.title("Best Pizza Capture & Quality Analysis")

    image_path = "captured_pizza_images/best_pizza.jpg"
    if not os.path.exists(image_path):
        st.warning("No pizza has been captured yet.")
        return

    # Show image and classification/analysis side by side
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image_path, caption="Best Detected Pizza", use_column_width=True)

    with col2:
        with st.spinner("Classifying pizza..."):
            label = classify_pizza(image_path)
        st.markdown("### Classification Result")
        st.markdown(f"**Prediction:** {label}")
        # Optionally, you can show confidence if your classify_pizza returns it

        st.markdown("---")
        with st.spinner("Analyzing pizza with Gemini..."):
            try:
                analysis = generate_gemini_analysis(image_path, label)
                st.markdown("### Gemini Quality Analysis")
                st.markdown(analysis)
            except Exception as e:
                st.error(f"Gemini analysis failed: {e}")

    # Add extra space and a divider for clarity
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

if __name__ == "__main__":
    main()

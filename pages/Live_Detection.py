# pages/Live_Detection.py

import streamlit as st
import cv2
import tempfile
import numpy as np
from modules.utils import detection_model, is_full_pizza, classify_pizza


def main():

    st.set_page_config(page_title="Pizza QC System", layout="wide")

    st.title("Realtime Pizza Monitoring")


    # Webcam capture
    run_detection = st.button("Start Webcam Detection")
    stop_detection = st.button("Stop")

    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    if run_detection:
        cap = cv2.VideoCapture(0)
        detected = False

        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Show spinner while model runs
            with st.spinner("Detecting pizza..."):
                results = detection_model(frame_rgb)

            best_pizza = None
            best_confidence = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = detection_model.names[class_id].lower()
                    if class_name == "pizza":
                        confidence = box.conf[0].item()
                        frame_height, frame_width = frame.shape[:2]
                        if is_full_pizza(box, frame_width, frame_height) and confidence > best_confidence:
                            best_pizza = box
                            best_confidence = confidence

            # Draw bounding box if pizza detected
            if best_pizza is not None:
                x1, y1, x2, y2 = map(int, best_pizza.xyxy[0])
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Pizza: {best_confidence:.2f}"
                cv2.putText(frame_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                detected = True
            else:
                detected = False

            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # If pizza detected, classify and show result
            if detected:
                # Save temp image for classification
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
                    cv2.imwrite(tmpfile.name, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    tmp_image_path = tmpfile.name

                with st.spinner("Classifying pizza..."):
                    label = classify_pizza(tmp_image_path)

                # Layout: Image and classification result side by side
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(tmp_image_path, caption="Detected Pizza", use_column_width=True)
                with col2:
                    st.markdown("### Classification Result")
                    st.markdown(f"**Prediction:** {label}")
                    st.markdown(f"**Confidence:** {best_confidence:.2%}")
                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()
                break  # Stop after first detection/classification

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2
import os
import tensorflow as tf
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
from dot_env import load_env

load_env()
# Initialize models once
detection_model = YOLO(r"C:\Users\Ishant Saraswat\Desktop\Project Pizza Classification\runs\detect\train12\weights\best.pt")
classification_model = tf.keras.models.load_model(
    r"C:\Users\Ishant Saraswat\Desktop\Project Pizza Classification\efficientnet_pizza_classifier.h5"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def get_reolink_stream(channel=1, stream_type='main'):
# """Generate RTSP URL for Reolink cameras"""
  return f"rtsp://admin:password@192.168.1.100:554/Preview_{channel:02d}_{stream_type}"

def is_full_pizza(box, frame_width, frame_height):
    """
    Check if the detected pizza covers at least 30% of the frame
    and has a near-square aspect ratio (close to a circle).
    
    Args:
        box: A bounding box object with .xyxy attribute (tensor/list/array of [x1, y1, x2, y2]).
        frame_width: Width of the image/frame.
        frame_height: Height of the image/frame.
        
    Returns:
        bool: True if the pizza meets area and aspect ratio criteria, else False.
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # [x1, y1, x2, y2]
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    frame_area = frame_width * frame_height
    aspect_ratio = box_width / float(box_height) if box_height != 0 else 0

    # Pizza must cover at least 30% of the frame and be roughly square (aspect ratio ~1)
    return (box_area >= 0.3 * frame_area) and (0.85 <= aspect_ratio <= 1.15)


def process_frame(frame):
# """YOLO detection with pizza area filtering"""
  results = detection_model(frame)
  for result in results:
    for box in result.boxes:
      if detection_model.names[int(box.cls)].lower() == "pizza":
        if is_full_pizza(box, frame.shape, frame.shape):
          return frame, box.conf.item()
  return None, 0

def classify_pizza(image_path):
# """EfficientNet classification"""
  img = cv2.imread(image_path)
  img = cv2.resize(img, (224, 224)) / 255.0
  prediction = classification_model.predict(np.expand_dims(img, axis=0))
  return "Good" if prediction > 0.5 else "Bad"

def generate_gemini_analysis(image_path, classification):
# """Multimodal analysis with Gemini"""
  model = genai.GenerativeModel("gemini-1.5-flash")
  with open(image_path, "rb") as img_file:
    return model.generate_content([
    f"Classification: {classification}. Analyze pizza quality based on:",
    "1. Bubbles 2. Cheese 3. Toppings 4. Burnt areas 5. Shape",
    Image.open(io.BytesIO(img_file.read()))
    ]).text
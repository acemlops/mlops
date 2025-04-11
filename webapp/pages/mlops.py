import sys
import os

# Add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import streamlit as st
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import load_model
from glob import glob
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras
from PIL import Image
import cv2
import base64

# Page Configuration 
st.set_page_config(page_title="Plant Disease Classification", layout="wide")
st.markdown("""
    <style>
    .reportview-container { background: #f5f5f5; }
    .stButton > button { width: 100%; border-radius: 10px; }
    .stImage { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# Load trained model
try:
    model = load_model("../models/trained.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Full class list (29 classes)
classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',    
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust_',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, [255, 255])
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def generate_gradcam(image, model):
    """ Dummy Grad-CAM visual (replace with real logic) """
    img_array = np.array(image)
    heatmap = np.uint8(255 * np.random.rand(*img_array.shape[:2]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return Image.fromarray(heatmap)

def download_report(pred_class, confidence):
    report_text = f""" 
    Plant Disease Classification Report
    -----------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="plant_disease_report.txt">Download Report</a>'
    return href

# Sidebar: Image Upload
st.sidebar.title("Upload Leaf Image")
uploaded_file = st.sidebar.file_uploader("Choose a plant leaf image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=False, width=400)

    if st.button("Classify Image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]

        # Safety check to avoid IndexError
        if pred_idx < len(classes):
            confidence = output[0][pred_idx] * 100
            pred_class = classes[pred_idx]
        else:
            pred_class = "Unknown"
            confidence = 0.0
            st.warning(f"Predicted index {pred_idx} is out of range for the classes list.")

        # Display Result
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")
        with col2:
            gradcam_image = generate_gradcam(image, model)
            st.image(gradcam_image, caption="Grad-CAM Visualisation", use_column_width=True)

        # Downloadable Report
        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)

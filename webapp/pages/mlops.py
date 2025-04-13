import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import io
import base64

# 🌿 Page config
st.set_page_config(page_title="🌿 Plant Disease Detection System", layout="centered")

# 🌿 Custom CSS for fullscreen dark glass box
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://static.vecteezy.com/system/resources/previews/026/489/428/large_2x/stunning-vertical-garden-backgrounds-featuring-lush-plants-against-a-captivating-wall-backdrop-perfect-for-adding-a-touch-of-nature-to-your-projects-ai-generated-photo.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 12px;
        margin-top: 120px;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0;
        height: 100%;
        width: 100%;
        background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.6));
        z-index: -1;
    }
    .glass-box h1, .glass-box h4, .glass-box p {
        color: white;
        text-align: center;
    }
    .glass-box .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .glass-box .stFileUploader {
        margin-bottom: 20px;
    }
    .stImage img {
        display: block;
        margin: 0 auto;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Load the model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/trained.h5")


try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ✅ 29 Class labels
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust_', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

# 🧊 BEGIN: Glass Box
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

# 🌿 Title
st.markdown('<h1>🌿 Plant Disease Detection System</h1>', unsafe_allow_html=True)

# 📤 Upload
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

# 🖼️ If image uploaded
if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("<h4>Original Image</h4>", unsafe_allow_html=True)
    st.image(image, width=300)

    if st.button("Predict Leaf Disease"):
        with st.spinner("🌿 Photosynthesizing..."):

            # Preprocessing
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_array = np.array(image)
            input_tensor = tf.image.resize(img_array, [255, 255]) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Predict
            preds = model.predict(input_tensor)
            class_idx = int(np.argmax(preds[0]))
            class_label = classes[class_idx]
            confidence = float(preds[0][class_idx]) * 100

            # 🔍 Result
            st.markdown(f"<h4>✅ Predicted Disease: <b>{class_label}</b></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>🔬 Confidence: <b>{confidence:.2f}%</b></h4>", unsafe_allow_html=True)

            if "healthy" in class_label.lower():
                st.success("✅ Your plant appears healthy!")
            else:
                st.error(f"⚠️ Detected disease: {class_label}")

            # 🧾 Generate report
            def generate_report(image, disease, confidence):
                report_img = image.copy()
                draw = ImageDraw.Draw(report_img)

                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                text = f"Disease: {disease}\nConfidence: {confidence:.2f}%"
                draw.text((10, 10), text, fill="white", font=font)

                buffer = io.BytesIO()
                report_img.save(buffer, format="PNG")
                buffer.seek(0)
                return buffer

            # 📥 Downloadable report
            report_buffer = generate_report(image, class_label, confidence)
            b64 = base64.b64encode(report_buffer.read()).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="plant_disease_report.png">📥 <b>Download Report</b></a>'
            st.markdown(href, unsafe_allow_html=True)

# 🧊 END: Glass Box
st.markdown('</div>', unsafe_allow_html=True)

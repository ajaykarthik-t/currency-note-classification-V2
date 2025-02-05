import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3
from PIL import Image
import pytesseract

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
except Exception as e:
    st.error(f"Text-to-speech initialization failed: {e}")
    engine = None

# Load the ML model
@st.cache_resource
def load_cached_model():
    try:
        return load_model('my_model.keras')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_cached_model()

# Preprocessing for OCR
def preprocess_image_for_ocr(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

# OCR-based text detection
def detect_text(img):
    processed_img = preprocess_image_for_ocr(img)
    text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 6', lang='eng+hin')
    return text.upper()

# ML-based currency prediction
def get_ml_prediction(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    return predicted_class, confidence

# Streamlit UI
st.title("Indian Currency Note Detection")
st.markdown("---")
option = st.radio("Choose Input Method:", ("Upload Image", "Use Camera"))

uploaded_img = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        uploaded_img = Image.open(uploaded_file)
elif option == "Use Camera":
    camera_img = st.camera_input("Take a photo of currency note")
    if camera_img:
        uploaded_img = Image.open(camera_img)

if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 OCR-Based Detection")
        with st.spinner("Extracting text..."):
            detected_text = detect_text(uploaded_img)
            if detected_text.strip():
                st.success("Text detected successfully!")
                st.text_area("Extracted Text:", detected_text, height=150)
            else:
                st.warning("No text detected. Try a clearer image.")
    
    with col2:
        st.subheader("🤖 ML-Based Detection")
        if model:
            with st.spinner("Analyzing image..."):
                ml_class, ml_confidence = get_ml_prediction(uploaded_img)
                st.success(f"Predicted Class: {ml_class} (Confidence: {ml_confidence:.2f})")
        else:
            st.error("ML Model not loaded properly.")

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import pyttsx3
from PIL import Image
import threading

# Initialize text-to-speech engine with error handling
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
except Exception as e:
    st.error(f"Text-to-speech initialization failed: {e}")
    engine = None

# Cache model loading for better performance
@st.cache_resource
def load_cached_model():
    try:
        return load_model('my_model.keras')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_cached_model()

# Define class-specific messages
class_messages = {
    0: "10 Rupees Note",
    1: "100 Rupees Note",
    2: "20 Rupees Note",
    3: "200 Rupees Note",
    4: "50 Rupees Note",
    5: "500 Rupees Note"
}

def process_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    return confidence, predicted_class

def speak_text(text):
    """Run text-to-speech in a separate thread to avoid Streamlit UI blocking."""
    if engine is not None:
        def speak():
            engine.say(text)
            engine.startLoop(False)  
            engine.endLoop()
        threading.Thread(target=speak).start()

st.title("🔊 Currency Detection for the Visually Impaired")
st.markdown("---")

# Speak an introduction message when the app starts
if "app_started" not in st.session_state:
    st.session_state.app_started = True
    speak_text("Welcome to Currency Detection for the Visually Impaired. Please upload an image or use the camera.")

# Use columns for better radio button layout
col1, col2 = st.columns(2)
with col1:
    option = st.radio("Choose Input Method:", ("Live Camera", "Upload Image"))

# Initialize session state variables
if 'last_spoken' not in st.session_state:
    st.session_state.last_spoken = ""

if option == "Live Camera":
    st.subheader("📷 Live Camera Detection")
    st.markdown("Position the currency note in the camera view and hold steady")
    
    camera_img = st.camera_input("Take a photo", label_visibility="hidden")
    
    if camera_img and model is not None:
        img = Image.open(camera_img)
        speak_text("Processing the image. Please wait.")
        
        with st.spinner("Analyzing..."):
            confidence, predicted_class = process_image(img)
        
        if confidence > 0.7:
            result_text = class_messages.get(predicted_class, "Unknown Note")
            st.success(f"**Detection Result:** {result_text} (Confidence: {confidence:.2f})")
            speak_text(result_text)
        else:
            st.warning("**Please position the note clearly in the frame**")
            speak_text("Could not detect the note. Please position it clearly.")

elif option == "Upload Image":
    st.subheader("📤 Upload Currency Note Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file is not None and model is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        speak_text("Image uploaded. Processing, please wait.")
        
        with st.spinner("Analyzing..."):
            confidence, predicted_class = process_image(img)
        
        if confidence > 0.7:
            result_text = class_messages.get(predicted_class, "Unknown Note")
            st.success(f"**Detection Result:** {result_text} (Confidence: {confidence:.2f})")
            speak_text(result_text)
        else:
            st.warning("**Could not recognize the note clearly. Please try another image.**")
            speak_text("Could not recognize the note clearly. Please try another image.")

# Add app footer with instructions
st.markdown("---")
st.markdown("**Instructions:** \n- Ensure good lighting \n- Position note flat in frame \n- Avoid glare and shadows")

# Add error message if model failed to load
if model is None:
    st.error("Application failed to initialize. Please check the model file and try again.")

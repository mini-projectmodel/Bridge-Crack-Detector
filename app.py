import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Configure web page
st.set_page_config(page_title="Bridge Crack Inspector", layout="centered")

st.title("Concrete Deck Crack Inspection System")
st.write("Upload a bridge surface photo to evaluate structural integrity.")

# Cache model in memory for fast inferencing
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('./model.h5')

model = load_trained_model()

# User file uploader
uploaded_file = st.file_uploader("Select an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Inspection Image', use_column_width=True)
    
    # Preprocess image to match training shape
    target_size = (224, 224)
    image_resized = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image_resized)
    img_batched = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    if st.button('Run Analysis'):
        prediction = model.predict(img_batched)
        confidence = float(prediction[0][0])
        
        if confidence < 0.5:
            crack_prob = (1 - confidence) * 100
            st.error(f"⚠️ **Defect Identified**: Structural Crack Detected (Confidence: {crack_prob:.2f}%)")
        else:
            clean_prob = confidence * 100
            st.success(f"✅ **Intact Surface**: No Cracks Detected (Confidence: {clean_prob:.2f}%)")

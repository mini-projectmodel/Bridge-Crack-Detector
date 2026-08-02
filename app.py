import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input  # <--- IMPORT THIS

# Configure web page
st.set_page_config(page_title="Bridge Crack Inspector", layout="centered")

st.title("Concrete Deck Crack Inspection System")
st.write("Upload a bridge surface photo to evaluate structural integrity.")

# Cache model in memory for fast inferencing
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('./new_model_2.h5')

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

    # --- THE FIX: Apply preprocessing here ---
    img_array = preprocess_input(img_array)  # <--- THIS LINE WAS MISSING

    img_batched = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    if st.button('Run Analysis'):
        prediction = model.predict(img_batched)

        # Note: Your model outputs a 2-class probability.
        # Check the output shape and adjust if necessary.
        # If your model output is [prob_cracked, prob_uncracked], this code is correct.
        crack_prob = float(prediction[0][0]) * 100
        clean_prob = float(prediction[0][1]) * 100

        if crack_prob > 50:
            st.error(f"⚠️ **Defect Identified**: Structural Crack Detected (Confidence: {crack_prob:.2f}%)")
        else:
            st.success(f"✅ **Intact Surface**: No Cracks Detected (Confidence: {clean_prob:.2f}%)")

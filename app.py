import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- NEW IMPORTS ---
# We will import the functions from our new utility file
# Make sure gradcam.py is in the same folder!
try:
    from gradcam import preprocess_uploaded_image, get_gradcam_heatmap, overlay_heatmap
except ImportError:
    st.error("Error: 'gradcam.py' not found. Please create it with the provided code.")
    st.stop()
except Exception as e:
    st.error(f"Error importing from gradcam.py: {e}")
    st.stop()


# --- Load the saved model ---
# (Your original code is perfect)
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('final_pneumonia_model_95_99_acc.keras')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model 'final_pneumonia_model_95_99_acc.keras': {e}")
        return None

model = load_my_model()

st.title("Pneumonia Detection from Chest X-Rays 🩺")
st.write("Upload a chest X-ray image, and the model will predict if it shows signs of pneumonia.")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None and model is not None:
    # --- Display the uploaded image ---
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # --- Preprocess the image (NEW, CLEANER WAY) ---
    # This one function now does all the work
    # 'preprocessed_img' goes to the model
    # 'original_img_array' is for display (0-255)
    preprocessed_img, original_img_array = preprocess_uploaded_image(uploaded_file)
    
    # --- Make prediction ---
    # (Uses our new preprocessed_img)
    prediction = model.predict(preprocessed_img)
    score = prediction[0][0]

    # --- Display the result ---
    # (Your original code)
    if score > 0.5:
        label = "PNEUMONIA"
        st.success(f"**Result: {label}** (Confidence: {score*100:.2f}%)")
    else:
        label = "NORMAL"
        st.success(f"**Result: {label}** (Confidence: {(1-score)*100:.2f}%)")

    # --- [NEW] GRAD-CAM VISUALIZATION SECTION ---
# ... existing code
    
    try:
        # 1. Generate the heatmap
        # --- [THE FIX] ---
        # We target "out_relu" again. This layer IS present
        # inside the 'mobilenetv2_1.00_224' layer, and the
        # tf-keras-vis library is smart enough to find it.
        heatmap = get_gradcam_heatmap(preprocessed_img, model, last_conv_layer_name="out_relu")
        
        if heatmap is not None:
            # 2. Overlay the heatmap on the original image
            superimposed_img, heatmap_only = overlay_heatmap(original_img_array, heatmap)

            # 3. Display the results in columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(heatmap_only, caption="AI Focus Heatmap", use_column_width=True)
            with col2:
                st.image(superimposed_img, caption="Heatmap Overlay", use_column_width=True)
            
            st.info("""
            **How to read this:** The highlighted (red/yellow) areas are what the AI 
            "looked at" most intensely to make its decision. This helps verify 
            if the model is focusing on the correct lung regions.
            """)
        
    except Exception as e:
        st.error(f"An error occurred while generating Grad-CAM: {e}")
        st.warning("This can happen if the model's layer names are different.")

elif uploaded_file is None:
    st.info("Please upload an image to begin.")

elif model is None:
    # This message is now part of the load_my_model() function
    pass
import tensorflow as tf
import numpy as np
import cv2  # This is OpenCV
import matplotlib.pyplot as plt
from PIL import Image

# --- NEW IMPORTS for tf-keras-vis ---
try:
    from tf_keras_vis.gradcam import Gradcam
    from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
    from tf_keras_vis.utils.scores import BinaryScore
except ImportError:
    # This will be caught by app.py, which is good.
    raise ImportError("Could not import tf-keras-vis. Please run: pip install tf-keras-vis")


def preprocess_uploaded_image(image_file, target_size=(224, 224)):
    """
    Loads and preprocesses an image file from Streamlit.
    (This function is correct and needs no changes)
    """
    # 1. Load the image from the uploaded file object
    img = Image.open(image_file).convert('RGB')
    
    # 2. Resize to the model's expected input
    img_resized = img.resize(target_size)
    
    # 3. Convert to a numpy array (this is (224, 224, 3) with values 0-255)
    original_img_array = np.array(img_resized)
    
    # 4. Expand dimensions to create a "batch" of 1
    preprocessed_img = np.expand_dims(original_img_array, axis=0)
    
    # Return the 0-255 preprocessed image and the 0-255 original
    return preprocessed_img.astype(np.float32), original_img_array.astype(np.uint8)

def get_gradcam_heatmap(img_array_0_255, model_with_preprocessing, last_conv_layer_name="out_relu"):
    """
    Generates the Grad-CAM heatmap by building a 100% clean
    model and transferring the weights.
    """
    
    # --- [THE DEFINITIVE WEIGHTS-TRANSFER SOLUTION] ---
    
    # 1. Manually preprocess the 0-255 image to the [-1, 1] range
    img_array_neg1_1 = tf.keras.applications.mobilenet_v2.preprocess_input(img_array_0_255)

    # 2. Build the "clean" model graph from scratch
    try:
        # 2a. Create a FRESH, CLEAN MobileNetV2 base
        # This has a clean graph, expecting [-1, 1] input.
        clean_base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights=None  # No weights yet
        )
        
        # 2b. Get the OLD layers from your loaded model
        old_base_model = model_with_preprocessing.get_layer('mobilenetv2_1.00_224')
        old_gap_layer = model_with_preprocessing.get_layer('global_average_pooling2d_1')
        old_dropout_layer = model_with_preprocessing.get_layer('dropout_1')
        old_dense_layer = model_with_preprocessing.get_layer('dense_1')

        # 2c. *** TRANSFER THE WEIGHTS ***
        clean_base_model.set_weights(old_base_model.get_weights())
        
        # 2d. Build the new model from the CLEAN base
        clean_input = clean_base_model.input
        x = clean_base_model.output
        x = old_gap_layer(x)  # It's fine to re-use these layers
        x = old_dropout_layer(x)
        output = old_dense_layer(x)
        
        # This is our new, 100% clean model
        gradcam_model = tf.keras.models.Model(clean_input, output)
        
    except Exception as e:
        raise ValueError(f"Failed to build the clean Grad-CAM model: {e}")

    # 3. Define the score (same as before)
    score = BinaryScore(1.0)

    # 4. Create Gradcam object using the *clean* model
    gradcam = Gradcam(gradcam_model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)
    
    # 5. Generate heatmap
    try:
        heatmap = gradcam(
            score,
            img_array_neg1_1,  # The [-1, 1] data
            penultimate_layer=last_conv_layer_name, # Target "out_relu"
        )
    except Exception as e:
        if 'Can not find layer' in str(e):
             raise ValueError(f"Could not find layer '{last_conv_layer_name}'. This is a critical build error.")
        raise e

    # 6. The library returns a list of heatmaps
    heatmap = heatmap[0]
    
    # 7. Normalize the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    return heatmap


def overlay_heatmap(original_img, heatmap, alpha=0.5, colormap_name='jet'):
    """
    Applies a color heatmap to an original image.
    (This function is perfect, no changes needed)
    """
    # 1. Resize the heatmap
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # 2. Get the colormap
    colormap = plt.get_cmap(colormap_name)
    
    # 3. Apply the colormap
    heatmap_colored = colormap(heatmap_resized)
    
    # 4. Convert to 0-255 RGB
    heatmap_colored_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # 5. Blend the two images
    superimposed_img = cv2.addWeighted(
        original_img, 1 - alpha, heatmap_colored_rgb, alpha, 0
    )
    
    return superimposed_img, heatmap_colored_rgb
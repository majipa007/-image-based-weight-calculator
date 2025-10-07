import streamlit as st
from PIL import Image
import os
import numpy as np
import tempfile # Import tempfile for secure temporary file handling
from segementer import segment_image, load_yolo_model # Import load_yolo_model
from depth_estimator import estimate_depth_heatmap, load_midas_model, calculate_goat_volume_and_weight_proxy
import matplotlib.pyplot as plt
import logging

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

st.set_page_config(layout="wide")
st.title("Goat Weight Detection and Depth Estimation")

# Define a target maximum side for image resizing
TARGET_MAX_SIDE = 640 # pixels

# Load MiDaS model once when the app starts
@st.cache_resource
def cached_load_midas_model():
    try:
        load_midas_model()
        return True
    except Exception as e:
        st.error(f"Failed to load depth estimation model: {e}")
        logging.error(f"Failed to load depth estimation model: {e}")
        return False

# Load YOLO model once when the app starts
@st.cache_resource
def cached_load_yolo_model():
    try:
        load_yolo_model()
        return True
    except Exception as e:
        st.error(f"Failed to load segmentation model: {e}")
        logging.error(f"Failed to load segmentation model: {e}")
        return False

# Check if both models loaded successfully
if not cached_load_midas_model():
    st.stop()
if not cached_load_yolo_model():
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    st.image(original_image, caption="Original Image", use_container_width=True)
    st.write("")

    width, height = original_image.size
    
    if width > height:
        ratio = TARGET_MAX_SIDE / width
        new_width = TARGET_MAX_SIDE
        new_height = int(height * ratio)
    else:
        ratio = TARGET_MAX_SIDE / height
        new_height = TARGET_MAX_SIDE
        new_width = int(width * ratio)

    resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    st.info(f"Original image {width}x{height}. Resized to {new_width}x{new_height} for consistent resolution.")

    # Use tempfile for robust temporary file handling
    temp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_image_path = temp_file.name
            resized_image.save(temp_image_path)
        
        col1, col2 = st.columns(2)

        segmentation_mask = None
        raw_depth_map = None

        with col1:
            st.write("### Segmentation")
            segmented_img, segmentation_mask = segment_image(temp_image_path)

            if segmented_img:
                st.image(segmented_img, caption="Segmented Image", use_container_width=True)
            else:
                st.write("No goats detected or an error occurred during segmentation.")
        
        with col2:
            st.write("### Depth Estimation")
            depth_heatmap_fig, raw_depth_map = estimate_depth_heatmap(temp_image_path)

            if depth_heatmap_fig:
                st.pyplot(depth_heatmap_fig)
                plt.close(depth_heatmap_fig)
            else:
                st.write("Could not estimate depth or an error occurred.")
        
        st.write("---")

        if segmentation_mask is not None and raw_depth_map is not None:
            st.write("### Volume and Weight Estimation (Proxy)")
            volume_proxy, weight_kg_proxy = calculate_goat_volume_and_weight_proxy(raw_depth_map, segmentation_mask)

            if volume_proxy is not None and weight_kg_proxy is not None:
                st.success(f"**Estimated Volume Proxy:** {volume_proxy:.2f} (arbitrary units)")
                st.success(f"**Estimated Weight Proxy:** {weight_kg_proxy:.2f} kg (approximate)")
                st.info("Note: These are proxy values due to the lack of camera calibration data and are highly approximate.")
            else:
                st.error("Could not calculate volume and weight proxies.")
        else:
            st.warning("Segmentation mask or raw depth map not available for volume/weight calculation.")
    
    finally:
        # Ensure temporary file is cleaned up
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logging.info(f"Cleaned up temporary file: {temp_image_path}")

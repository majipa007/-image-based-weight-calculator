import streamlit as st
from PIL import Image
import os
import numpy as np # Added for array operations
from segementer import segment_image
from depth_estimator import estimate_depth_heatmap, load_midas_model, calculate_goat_volume_and_weight_proxy
import matplotlib.pyplot as plt

st.set_page_config(layout="wide") # Use wide layout for better display of multiple images
st.title("Goat Weight Detection and Depth Estimation")

# Load MiDaS model once when the app starts
# This will run only once per session
@st.cache_resource
def cached_load_midas_model():
    try:
        load_midas_model()
        return True
    except Exception as e:
        st.error(f"Failed to load depth estimation model: {e}")
        return False

if not cached_load_midas_model():
    st.stop() # Stop the app if model loading fails

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_image_path = "temp_image.jpg"
    with open(os.path.join(temp_image_path), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Original Image", use_container_width=True)
    st.write("")

    col1, col2 = st.columns(2)

    # Initialize variables for mask and raw depth
    segmentation_mask = None
    raw_depth_map = None

    with col1:
        st.write("### Segmentation")
        # Perform segmentation
        segmented_img, segmentation_mask = segment_image(temp_image_path)

        if segmented_img:
            st.image(segmented_img, caption="Segmented Image", use_container_width=True)
        else:
            st.write("No goats detected or an error occurred during segmentation.")
    
    with col2:
        st.write("### Depth Estimation")
        # Perform depth estimation
        depth_heatmap_fig, raw_depth_map = estimate_depth_heatmap(temp_image_path)

        if depth_heatmap_fig:
            st.pyplot(depth_heatmap_fig)
            plt.close(depth_heatmap_fig) # Close the figure to free up memory
        else:
            st.write("Could not estimate depth or an error occurred.")
    
    st.write("---") # Separator

    # Calculate Volume and Weight Proxies
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
    
    # Clean up the temporary file
    os.remove(temp_image_path)

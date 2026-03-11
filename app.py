import streamlit as st
from PIL import Image
import os
import tempfile # Import tempfile for secure temporary file handling
import cv2
import numpy as np
from segementer import segment_image, load_yolo_model, track_goats_in_video
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

input_mode = st.radio("Select input type", ["Image", "Video"], horizontal=True)

st.sidebar.title("Postgres Settings")
db_host = st.sidebar.text_input("Host", value=os.getenv("PGHOST", "localhost"))
db_port = st.sidebar.number_input("Port", min_value=1, max_value=65535, value=int(os.getenv("PGPORT", "5432")))
db_name = st.sidebar.text_input("Database", value=os.getenv("PGDATABASE", "goat_weight"))
db_user = st.sidebar.text_input("User", value=os.getenv("PGUSER", "goat_user"))
db_password = st.sidebar.text_input("Password", value=os.getenv("PGPASSWORD", "goat_pass"), type="password")

def overlay_mask_on_image_bytes(image_jpg_bytes, mask_png_bytes, mask_color=(0, 255, 0), alpha=0.35):
    image_arr = np.frombuffer(image_jpg_bytes, dtype=np.uint8)
    mask_arr = np.frombuffer(mask_png_bytes, dtype=np.uint8)

    image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    mask_gray = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or mask_gray is None:
        return image_jpg_bytes

    mask_binary = mask_gray > 0
    overlay = np.zeros_like(image_bgr, dtype=np.uint8)
    overlay[mask_binary] = mask_color
    blended = cv2.addWeighted(image_bgr, 1.0, overlay, alpha, 0)

    ok, encoded = cv2.imencode(".jpg", blended)
    if not ok:
        return image_jpg_bytes
    return encoded.tobytes()

def process_image(uploaded_image_file):
    original_image = Image.open(uploaded_image_file)
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

def process_video(uploaded_video_file):
    conf_threshold = st.slider("Tracking confidence threshold", 0.1, 0.95, 0.5, 0.05)
    resize_scale = st.slider("Video resize scale", 0.2, 1.0, 0.4, 0.05)
    top_k = st.slider("Top masks used per goat (for final weight)", 3, 5, 3, 1)

    db_config = {
        "host": db_host,
        "port": int(db_port),
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
    }

    temp_input_path = None
    temp_output_path = None
    try:
        suffix = os.path.splitext(uploaded_video_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
            temp_input_path = temp_input.name
            temp_input.write(uploaded_video_file.getbuffer())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            temp_output_path = temp_output.name

        if st.button("Run Tracking + Weight Estimation", type="primary"):
            with st.spinner("Processing video... this can take some time."):
                output_path, run_id, summary, frames_processed = track_goats_in_video(
                    temp_input_path,
                    temp_output_path,
                    db_config=db_config,
                    source_name=uploaded_video_file.name,
                    top_k=top_k,
                    conf_threshold=conf_threshold,
                    scale=resize_scale,
                )

            if output_path is None:
                st.error("Video processing failed. Check Postgres settings and model loading.")
                return

            st.success(f"Processed {frames_processed} frames. Run ID: {run_id}")
            with open(output_path, "rb") as f:
                st.video(f.read())

            st.write("### Per-Goat Final Results")
            if summary:
                for item in summary:
                    goat_id = item["goat_id"]
                    final_weight = item["final_weight_proxy_kg"]
                    samples_used = item["samples_used"]
                    preview_samples = item.get("preview_samples", [])

                    st.write(f"#### Goat ID {goat_id}")
                    st.write(f"Final Weight Proxy: **{final_weight:.2f} kg** (from {samples_used} top-mask samples)")

                    img_cols = st.columns(3)
                    for idx, col in enumerate(img_cols):
                        if idx < len(preview_samples):
                            sample = preview_samples[idx]
                            overlaid = overlay_mask_on_image_bytes(
                                sample["image_jpg"],
                                sample["mask_png"],
                            )
                            col.image(overlaid, caption=f"Sample {idx + 1} (mask overlaid)", use_container_width=True)
                        else:
                            col.write("No image")
            else:
                st.warning("No tracked goat weights were produced.")

            st.info("Weight values shown in video mode are proxy estimates (not calibrated true weights).")
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            logging.info(f"Cleaned up temporary input video: {temp_input_path}")
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logging.info(f"Cleaned up temporary output video: {temp_output_path}")

if input_mode == "Image":
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_image is not None:
        process_image(uploaded_image)
else:
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    if uploaded_video is not None:
        process_video(uploaded_video)

import logging
import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from depth_estimator import (
    calculate_goat_volume_and_weight_proxy,
    estimate_depth_heatmap,
    load_midas_model,
)
from segementer import (
    detect_goats_in_image,
    load_yolo_model,
    process_video_with_detection,
    segment_image,
)

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

st.set_page_config(layout="wide")
st.title("Goat Analysis: Segmentation, Depth & Detection")

# Define a target maximum side for image resizing
TARGET_MAX_SIDE = 640  # pixels


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
        st.error(f"Failed to load detection/segmentation model: {e}")
        logging.error(f"Failed to load detection/segmentation model: {e}")
        return False


# Check if both models loaded successfully
if not cached_load_midas_model():
    st.stop()
if not cached_load_yolo_model():
    st.stop()

mode = st.sidebar.selectbox(
    "Select mode",
    (
        "Image: Segmentation + Depth + Weight Proxy",
        "Image: Object Detection only",
        "Video: Object Detection",
    ),
)

if mode == "Image: Segmentation + Depth + Weight Proxy":
    uploaded_file = st.file_uploader(
        "Choose an image for segmentation and depth...", type=["jpg", "jpeg", "png"]
    )

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

        resized_image = original_image.resize(
            (new_width, new_height), Image.LANCZOS
        )
        st.info(
            f"Original image {width}x{height}. Resized to "
            f"{new_width}x{new_height} for consistent resolution."
        )

        # Use tempfile for robust temporary file handling
        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jpg"
            ) as temp_file:
                temp_image_path = temp_file.name
                resized_image.save(temp_image_path)

            col1, col2 = st.columns(2)

            segmentation_mask = None
            raw_depth_map = None

            with col1:
                st.write("### Segmentation")
                segmented_img, segmentation_mask = segment_image(temp_image_path)

                if segmented_img:
                    st.image(
                        segmented_img,
                        caption="Segmented Image",
                        use_container_width=True,
                    )
                else:
                    st.write(
                        "No goats detected or an error occurred during segmentation."
                    )

            with col2:
                st.write("### Depth Estimation")
                depth_heatmap_fig, raw_depth_map = estimate_depth_heatmap(
                    temp_image_path
                )

                if depth_heatmap_fig:
                    st.pyplot(depth_heatmap_fig)
                    plt.close(depth_heatmap_fig)
                else:
                    st.write("Could not estimate depth or an error occurred.")

            st.write("---")

            if segmentation_mask is not None and raw_depth_map is not None:
                st.write("### Volume and Weight Estimation (Proxy)")
                (
                    volume_proxy,
                    weight_kg_proxy,
                ) = calculate_goat_volume_and_weight_proxy(
                    raw_depth_map, segmentation_mask
                )

                if volume_proxy is not None and weight_kg_proxy is not None:
                    st.success(
                        f"**Estimated Volume Proxy:** {volume_proxy:.2f} "
                        "(arbitrary units)"
                    )
                    st.success(
                        f"**Estimated Weight Proxy:** {weight_kg_proxy:.2f} kg "
                        "(approximate)"
                    )
                    st.info(
                        "Note: These are proxy values due to the lack of camera "
                        "calibration data and are highly approximate."
                    )
                else:
                    st.error("Could not calculate volume and weight proxies.")
            else:
                st.warning(
                    "Segmentation mask or raw depth map not available for "
                    "volume/weight calculation."
                )

        finally:
            # Ensure temporary file is cleaned up
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                logging.info(f"Cleaned up temporary file: {temp_image_path}")

elif mode == "Image: Object Detection only":
    uploaded_file = st.file_uploader(
        "Choose an image for object detection...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Image", use_container_width=True)

        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jpg"
            ) as temp_file:
                temp_image_path = temp_file.name
                original_image.save(temp_image_path)

            annotated_img, detections = detect_goats_in_image(temp_image_path)

            if annotated_img is not None:
                st.image(
                    annotated_img,
                    caption="Object Detection (Goats)",
                    use_container_width=True,
                )

                if detections:
                    st.write("### Detections")
                    for idx, det in enumerate(detections, start=1):
                        x1, y1, x2, y2 = det["box"]
                        st.write(
                            f"Goat {idx}: Box=({x1}, {y1}, {x2}, {y2}), "
                            f"Confidence={det['conf']:.2f}"
                        )
                else:
                    st.info("No goats detected above the confidence threshold.")
            else:
                st.error("Detection failed for this image.")
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                logging.info(f"Cleaned up temporary file: {temp_image_path}")
elif mode == "Video: Object Detection":
    video_file = st.file_uploader(
        "Upload a video for goat detection...",
        type=["mp4", "avi", "mov", "mkv"],
    )

    enable_weight = st.checkbox(
        "Enable weight estimation (samples every 15 frames — slower)",
        value=False,
    )

    if video_file is not None:
        input_video_path = None
        output_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_in:
                input_video_path = temp_in.name
                temp_in.write(video_file.read())

            output_video_path = input_video_path.replace(".mp4", "_annotated.mp4")

            st.info("Processing video, please wait...")
            success, weight_estimates = process_video_with_detection(
                input_video_path,
                output_video_path,
                estimate_weight=enable_weight,
                weight_sample_interval=15,
                estimate_depth_fn=estimate_depth_heatmap if enable_weight else None,
                calc_weight_fn=calculate_goat_volume_and_weight_proxy if enable_weight else None,
            )

            if success and os.path.exists(output_video_path):
                st.video(output_video_path)

                # Show weight summary if estimation was enabled
                if enable_weight and weight_estimates:
                    avg_weight = sum(weight_estimates) / len(weight_estimates)
                    st.success(f"**Average estimated weight:** {avg_weight:.2f} kg (proxy, ~{len(weight_estimates)} samples)")
                    st.info(
                        "Note: Weight values are proxy estimates based on depth + segmentation "
                        "without camera calibration. They are approximate."
                    )
                elif enable_weight:
                    st.warning("Weight estimation was enabled but no valid estimates were produced. "
                               "Check that goats are visible and segmentation is working.")

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download annotated video",
                        data=f,
                        file_name="annotated_goat_detection.mp4",
                        mime="video/mp4",
                    )
            else:
                st.error("Failed to process video.")
        finally:
            if input_video_path and os.path.exists(input_video_path):
                os.remove(input_video_path)
                logging.info(f"Cleaned up temporary file: {input_video_path}")
            if output_video_path and os.path.exists(output_video_path):
                logging.info(f"Annotated video available at: {output_video_path}")
import streamlit as st
from PIL import Image
import os
import tempfile  # Import tempfile for secure temporary file handling
import cv2
import numpy as np
from urllib.parse import urlparse
from segementer import segment_image, load_yolo_model, track_goats_in_video
from depth_estimator import (
    estimate_depth_heatmap,
    load_midas_model,
    calculate_goat_volume_and_weight_proxy,
)
import matplotlib.pyplot as plt
import logging

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

st.set_page_config(layout="wide")
st.title("Goat Weight Detection and Depth Estimation")

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
db_port = st.sidebar.number_input(
    "Port", min_value=1, max_value=65535, value=int(os.getenv("PGPORT", "5432"))
)
db_name = st.sidebar.text_input(
    "Database", value=os.getenv("PGDATABASE", "goat_weight")
)
db_user = st.sidebar.text_input("User", value=os.getenv("PGUSER", "goat_user"))
db_password = st.sidebar.text_input(
    "Password", value=os.getenv("PGPASSWORD", "goat_pass"), type="password"
)


def overlay_mask_on_image_bytes(
    image_jpg_bytes, mask_png_bytes, mask_color=(0, 255, 0), alpha=0.35
):
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


def run_segmentation_depth_and_weight(
    original_image, original_caption="Original Image"
):
    st.image(original_image, caption=original_caption, use_container_width=True)
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
        (new_width, new_height), Image.Resampling.LANCZOS
    )
    st.info(
        f"Original image {width}x{height}. Resized to {new_width}x{new_height} for consistent resolution."
    )

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
                st.image(
                    segmented_img, caption="Segmented Image", use_container_width=True
                )
            else:
                st.write("No goats detected or an error occurred during segmentation.")

        with col2:
            st.write("### Depth Estimation")
            depth_result = estimate_depth_heatmap(temp_image_path)
            if depth_result is not None:
                depth_heatmap_fig, raw_depth_map = depth_result
            else:
                depth_heatmap_fig, raw_depth_map = None, None

            if depth_heatmap_fig:
                st.pyplot(depth_heatmap_fig)
                plt.close(depth_heatmap_fig)
            else:
                st.write("Could not estimate depth or an error occurred.")

        st.write("---")

        if segmentation_mask is not None and raw_depth_map is not None:
            st.write("### Volume and Weight Estimation (Proxy)")
            volume_proxy, weight_kg_proxy = calculate_goat_volume_and_weight_proxy(
                raw_depth_map, segmentation_mask
            )

            if volume_proxy is not None and weight_kg_proxy is not None:
                st.success(
                    f"**Estimated Volume Proxy:** {volume_proxy:.2f} (arbitrary units)"
                )
                st.success(
                    f"**Estimated Weight Proxy:** {weight_kg_proxy:.2f} kg (approximate)"
                )
                st.info(
                    "Note: These are proxy values due to the lack of camera calibration data and are highly approximate."
                )
            else:
                st.error("Could not calculate volume and weight proxies.")
        else:
            st.warning(
                "Segmentation mask or raw depth map not available for volume/weight calculation."
            )

    finally:
        # Ensure temporary file is cleaned up
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            logging.info(f"Cleaned up temporary file: {temp_image_path}")


def process_image(uploaded_image_file):
    original_image = Image.open(uploaded_image_file)
    run_segmentation_depth_and_weight(original_image, original_caption="Original Image")


def get_video_metadata(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 0 or total_frames is None or total_frames <= 0:
        return {
            "fps": None,
            "total_frames": None,
            "duration_seconds": None,
        }

    return {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "duration_seconds": float(total_frames / fps),
    }


def extract_frame_at_timestamp(video_source, timestamp_seconds):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return None, "Could not open video source for frame extraction."

    if timestamp_seconds < 0:
        cap.release()
        return None, "Timestamp must be zero or greater."

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    success, frame_bgr = cap.read()

    if not success:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and fps > 0:
            frame_index = int(round(timestamp_seconds * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
            success, frame_bgr = cap.read()

    cap.release()

    if not success or frame_bgr is None:
        return None, "Could not decode a frame at the selected timestamp."

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, None


def process_single_frame_video_mode():
    st.write("### Single-Frame Processing")
    st.caption(
        "Play the video, pause at your moment, then use timestamp-based frame processing."
    )

    source_type = st.radio(
        "Video source",
        ["Upload file", "URL"],
        horizontal=True,
        key="single_frame_source_type",
    )

    temp_uploaded_video_path = None
    video_source = None
    selected_timestamp = 0.0

    try:
        if source_type == "Upload file":
            uploaded_video = st.file_uploader(
                "Choose a video...",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                key="single_frame_video_uploader",
            )

            if uploaded_video is None:
                return

            video_bytes = uploaded_video.getvalue()
            st.video(video_bytes)

            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_uploaded_video_path = temp_file.name
                temp_file.write(video_bytes)

            video_source = temp_uploaded_video_path
        else:
            video_url = st.text_input(
                "Video URL",
                placeholder="https://example.com/video.mp4",
                key="single_frame_video_url",
            ).strip()

            if not video_url:
                return

            st.video(video_url)
            video_source = video_url

            parsed = urlparse(video_url)
            host = parsed.netloc.lower()
            if "youtube.com" in host or "youtu.be" in host:
                st.info(
                    "YouTube URLs can play in the app, but frame extraction works only for direct video files or uploaded videos."
                )

        metadata = get_video_metadata(video_source)
        if metadata is None:
            st.warning(
                "Could not read video metadata. You can still try processing with a manual timestamp."
            )
            selected_timestamp = st.number_input(
                "Timestamp to process (seconds)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.2f",
                key="single_frame_timestamp_number_unknown",
            )
        else:
            duration_seconds = metadata["duration_seconds"]
            fps = metadata["fps"]
            total_frames = metadata["total_frames"]

            if duration_seconds is not None:
                st.caption(
                    f"Duration: {duration_seconds:.2f}s | FPS: {fps:.2f} | Frames: {total_frames}"
                )
                max_slider = max(float(duration_seconds), 0.1)
                selected_timestamp = st.slider(
                    "Timestamp to process (seconds)",
                    min_value=0.0,
                    max_value=max_slider,
                    value=0.0,
                    step=0.1,
                    key="single_frame_timestamp_slider",
                )
            else:
                st.caption(
                    "Video metadata is partially unavailable; use a manual timestamp."
                )
                selected_timestamp = st.number_input(
                    "Timestamp to process (seconds)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key="single_frame_timestamp_number_partial",
                )

        if st.button(
            "Process Selected Frame", type="primary", key="single_frame_process_button"
        ):
            with st.spinner(
                "Extracting frame and running segmentation/depth/weight estimation..."
            ):
                frame_rgb, extraction_error = extract_frame_at_timestamp(
                    video_source, selected_timestamp
                )

            if extraction_error is not None:
                st.error(extraction_error)
                return

            if frame_rgb is None:
                st.error("Frame extraction returned no data.")
                return

            selected_frame = Image.fromarray(frame_rgb)
            run_segmentation_depth_and_weight(
                selected_frame,
                original_caption=f"Selected Frame @ {selected_timestamp:.2f}s",
            )

    finally:
        if temp_uploaded_video_path and os.path.exists(temp_uploaded_video_path):
            os.remove(temp_uploaded_video_path)
            logging.info(
                f"Cleaned up temporary uploaded video: {temp_uploaded_video_path}"
            )


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
                st.error(
                    "Video processing failed. Check Postgres settings and model loading."
                )
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
                    st.write(
                        f"Final Weight Proxy: **{final_weight:.2f} kg** (from {samples_used} top-mask samples)"
                    )

                    img_cols = st.columns(3)
                    for idx, col in enumerate(img_cols):
                        if idx < len(preview_samples):
                            sample = preview_samples[idx]
                            overlaid = overlay_mask_on_image_bytes(
                                sample["image_jpg"],
                                sample["mask_png"],
                            )
                            col.image(
                                overlaid,
                                caption=f"Sample {idx + 1} (mask overlaid)",
                                use_container_width=True,
                            )
                        else:
                            col.write("No image")
            else:
                st.warning("No tracked goat weights were produced.")

            st.info(
                "Weight values shown in video mode are proxy estimates (not calibrated true weights)."
            )
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            logging.info(f"Cleaned up temporary input video: {temp_input_path}")
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            logging.info(f"Cleaned up temporary output video: {temp_output_path}")


if input_mode == "Image":
    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader"
    )
    if uploaded_image is not None:
        process_image(uploaded_image)
else:
    tracking_tab, single_frame_tab = st.tabs(
        ["Tracking + Weight", "Single Frame From Video"]
    )

    with tracking_tab:
        uploaded_video = st.file_uploader(
            "Choose a video...", type=["mp4", "mov", "avi", "mkv"], key="video_uploader"
        )
        if uploaded_video is not None:
            process_video(uploaded_video)

    with single_frame_tab:
        process_single_frame_video_mode()

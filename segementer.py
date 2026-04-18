from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging
from collections import defaultdict, deque
from depth_estimator import (
    estimate_depth_map_from_rgb,
    calculate_goat_volume_and_weight_proxy,
)
from db_store import create_store_from_config

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global YOLO model to avoid reloading on every function call
yolo_model = None


def load_yolo_model(model_path="model.pt"):
    """Loads the YOLO segmentation model."""
    global yolo_model
    if yolo_model is None:
        logging.info(f"Loading YOLO model from {model_path}...")
        try:
            yolo_model = YOLO(model_path)
            logging.info("YOLO model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading YOLO model: {e}")
            yolo_model = None
            raise  # Re-raise the exception to indicate failure


def segment_image(image_path):
    """
    Performs segmentation on an input image using a YOLOv8n-seg model.
    Args:
        image_path (str): Path to the input image.
    Returns:
        PIL.Image.Image: Annotated image with segmentation masks.
        np.array: Binary segmentation mask.
    """
    if yolo_model is None:
        try:
            load_yolo_model()
        except Exception:
            logging.error("YOLO model not loaded, cannot perform segmentation.")
            return None, None

    try:
        logging.info(f"Segmentation started for {image_path}...")
        results = yolo_model(image_path)  # Predict on the image

        segmented_img_pil = None
        segmentation_mask = None

        for result in results:
            # Get the plotted image without any text or labels
            img_array = result.plot(labels=False, boxes=False)

            # Convert to OpenCV format for further processing
            img_cv = img_array.copy()

            if result.masks is not None and len(result.masks.data) > 0:
                # Combine all masks into a single binary mask
                mask_data_np = result.masks.data.cpu().numpy()
                original_h, original_w, _ = img_array.shape
                combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)

                for mask_single in mask_data_np:
                    mask_resized = cv2.resize(
                        mask_single,
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    combined_mask = np.maximum(combined_mask, mask_resized)

                segmentation_mask = combined_mask
            else:
                logging.info(f"No masks detected for {image_path}.")

            segmented_img_pil = Image.fromarray(
                img_cv[..., ::-1]
            )  # Convert BGR to RGB for PIL
            break  # Process only the first result for simplicity

        logging.info("Segmentation completed.")
        return segmented_img_pil, segmentation_mask

    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return None, None


def _decode_bytes_to_bgr(image_bytes):
    image_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_arr, cv2.IMREAD_COLOR)


def _decode_bytes_to_mask(mask_bytes):
    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    decoded_mask = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
    if decoded_mask is None:
        return None
    return (decoded_mask > 0).astype(np.uint8)


def track_goats_in_video(
    video_path,
    output_path,
    db_config,
    source_name,
    top_k=3,
    conf_threshold=0.5,
    scale=0.4,
    tail_length=15,
    tail_color=(0, 255, 0),
    tail_thickness=3,
    mask_color=(0, 255, 0),
    mask_alpha=0.35,
    orientation_mode="landscape",
):
    """
    Tracks goats in a video, stores candidate goat crops/masks in Postgres,
    and computes final per-goat proxy weight using top-K mask areas.
    Args:
        video_path (str): Input video path.
        output_path (str): Output path for annotated video.
        db_config (dict): Postgres connection settings.
        source_name (str): Human-readable source video name.
        top_k (int): Number of top masks per goat used for final weight estimation.
        conf_threshold (float): YOLO confidence threshold.
        scale (float): Resize scale applied before inference.
        tail_length (int): Number of historical center points for motion tails.
        orientation_mode (str): "landscape" for default, "portrait" to rotate frames 90 deg.
    Returns:
        tuple: (output_path, run_id, per_goat_summary, frames_processed) or
               (None, None, None, 0) on failure.
    """
    store = create_store_from_config(db_config)
    if store is None:
        return None, None, None, 0

    if yolo_model is None:
        try:
            load_yolo_model()
        except Exception:
            logging.error("YOLO model not loaded, cannot perform video tracking.")
            store.close()
            return None, None, None, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        store.close()
        return None, None, None, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate_to_portrait = str(orientation_mode).lower() == "portrait"
    base_w = orig_h if rotate_to_portrait else orig_w
    base_h = orig_w if rotate_to_portrait else orig_h
    new_w = max(1, int(base_w * scale))
    new_h = max(1, int(base_h * scale))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (new_w, new_h),
    )

    track_history = defaultdict(lambda: deque(maxlen=tail_length))
    track_id_to_short_id = {}
    next_short_id = 1
    frames_processed = 0
    run_id = None
    tracking_error = False

    logging.info(f"Video tracking started for {video_path}...")

    try:
        run_id = store.create_video_run(source_name=source_name, top_k=top_k)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1
            if rotate_to_portrait:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            results = yolo_model.track(
                frame, conf=conf_threshold, persist=True, verbose=False
            )
            result = results[0]
            annotated_frame = frame.copy()
            overlay = np.zeros_like(annotated_frame, dtype=np.uint8)

            has_tracks = (
                result.boxes is not None
                and result.boxes.id is not None
                and len(result.boxes.id) > 0
            )
            has_masks = result.masks is not None and len(result.masks.data) > 0

            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            track_ids = result.boxes.id.cpu().numpy().astype(int) if has_tracks else []
            masks = result.masks.data.cpu().numpy() if has_masks else []

            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                if track_id not in track_id_to_short_id:
                    track_id_to_short_id[track_id] = next_short_id
                    next_short_id += 1
                short_id = track_id_to_short_id[track_id]

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                track_history[track_id].append((cx, cy))

                if has_masks and idx < len(masks):
                    mask = cv2.resize(
                        masks[idx].astype(np.float32),
                        (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    overlay[binary_mask > 0] = mask_color

                    x1_c = max(0, min(new_w - 1, x1))
                    y1_c = max(0, min(new_h - 1, y1))
                    x2_c = max(0, min(new_w, x2))
                    y2_c = max(0, min(new_h, y2))

                    if x2_c > x1_c and y2_c > y1_c:
                        crop_image = frame[y1_c:y2_c, x1_c:x2_c]
                        crop_mask = binary_mask[y1_c:y2_c, x1_c:x2_c]
                        mask_area = int(np.sum(crop_mask))

                        if mask_area > 0 and crop_image.size > 0 and crop_mask.size > 0:
                            ok_img, img_buf = cv2.imencode(".jpg", crop_image)
                            ok_mask, mask_buf = cv2.imencode(
                                ".png", (crop_mask * 255).astype(np.uint8)
                            )
                            if ok_img and ok_mask:
                                store.add_candidate(
                                    run_id=run_id,
                                    goat_id=short_id,
                                    frame_index=frames_processed,
                                    mask_area=mask_area,
                                    crop_image_jpg=img_buf.tobytes(),
                                    mask_png=mask_buf.tobytes(),
                                )

                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    cv2.line(
                        annotated_frame,
                        pts[i - 1],
                        pts[i],
                        tail_color,
                        tail_thickness,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    annotated_frame,
                    f"ID {short_id}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    tail_color,
                    2,
                    cv2.LINE_AA,
                )

            annotated_frame = cv2.addWeighted(
                annotated_frame, 1.0, overlay, mask_alpha, 0
            )
            writer.write(annotated_frame)
    except Exception as e:
        logging.error(f"Error during video tracking: {e}")
        tracking_error = True
    finally:
        try:
            store.flush_candidates()
        except Exception as flush_error:
            logging.error(f"Failed to flush candidates: {flush_error}")
            tracking_error = True
        cap.release()
        writer.release()

    if tracking_error or run_id is None:
        store.close()
        return None, None, None, frames_processed

    top_candidates = store.fetch_top_candidates(run_id=run_id, top_k=top_k)
    summary = []
    for goat_id in sorted(top_candidates.keys()):
        candidates = top_candidates[goat_id]
        if not candidates:
            continue

        weights = []
        preview_samples = []
        for candidate in candidates:
            bgr_crop = _decode_bytes_to_bgr(candidate["crop_image_jpg"])
            mask_crop = _decode_bytes_to_mask(candidate["mask_png"])

            if bgr_crop is None or mask_crop is None:
                continue

            rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            raw_depth_map = estimate_depth_map_from_rgb(rgb_crop)
            if raw_depth_map is None:
                continue

            _, weight_kg_proxy = calculate_goat_volume_and_weight_proxy(
                raw_depth_map, mask_crop
            )
            if weight_kg_proxy is None:
                continue

            weights.append(float(weight_kg_proxy))
            preview_samples.append(
                {
                    "image_jpg": candidate["crop_image_jpg"],
                    "mask_png": candidate["mask_png"],
                }
            )

        if not weights:
            continue

        final_weight = float(np.mean(weights))
        store.upsert_goat_result(
            run_id=run_id,
            goat_id=goat_id,
            weight_proxy_kg=final_weight,
            samples_used=len(weights),
        )

        summary.append(
            {
                "goat_id": goat_id,
                "final_weight_proxy_kg": final_weight,
                "samples_used": len(weights),
                "preview_samples": preview_samples[:3],
            }
        )

    store.close()
    logging.info(f"Video tracking completed. Frames processed: {frames_processed}")
    return output_path, run_id, summary, frames_processed

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging
import tempfile
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
            raise


def segment_image(image_path):
    """
    Performs segmentation on an input image using a YOLOv8n-seg model.
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
        results = yolo_model(image_path)

        segmented_img_pil = None
        segmentation_mask = None

        for result in results:
            img_array = result.plot(labels=False, boxes=False)
            img_cv = img_array.copy()

            if result.masks is not None and len(result.masks.data) > 0:
                mask_data_np = result.masks.data.cpu().numpy()
                original_h, original_w, _ = img_array.shape
                combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)

                for mask_single in mask_data_np:
                    mask_resized = cv2.resize(
                        mask_single, (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    combined_mask = np.maximum(combined_mask, mask_resized)

                segmentation_mask = combined_mask
            else:
                logging.info(f"No masks detected for {image_path}.")

            segmented_img_pil = Image.fromarray(img_cv[..., ::-1])
            break

        logging.info("Segmentation completed.")
        return segmented_img_pil, segmentation_mask

    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return None, None


def detect_goats_in_image(image_path, conf_threshold=0.3):
    """
    Runs object detection on an image and returns annotated image plus detections.
    """
    if yolo_model is None:
        try:
            load_yolo_model()
        except Exception:
            logging.error("YOLO model not loaded, cannot perform detection.")
            return None, []

    try:
        logging.info(f"Object detection started for {image_path}...")
        results = yolo_model(image_path)

        annotated_pil = None
        detections = []

        for result in results:
            img_array = result.plot(labels=True, boxes=True, masks=False)

            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, classes):
                    if conf < conf_threshold:
                        continue
                    detections.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": float(conf),
                        "cls": cls_id,
                    })

            annotated_pil = Image.fromarray(img_array[..., ::-1])
            break

        logging.info(f"Object detection completed with {len(detections)} detections.")
        return annotated_pil, detections

    except Exception as e:
        logging.error(f"Error during object detection: {e}")
        return None, []


def _apply_green_mask_overlay(frame_bgr, seg_mask, alpha=0.45):
    """
    Blends a semi-transparent green mask onto the frame wherever seg_mask > 0.
    Also draws a white contour around the mask edge.
    Args:
        frame_bgr (np.array): BGR video frame.
        seg_mask (np.array): Binary mask (H x W), values 0 or 1.
        alpha (float): Opacity of the green overlay.
    Returns:
        np.array: Frame with green overlay applied.
    """
    overlay = frame_bgr.copy()
    green = np.zeros_like(frame_bgr)
    green[:, :] = (0, 255, 0)  # BGR green

    mask_bool = seg_mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * frame_bgr[mask_bool] + alpha * green[mask_bool]
    ).astype(np.uint8)

    # White contour around mask boundary
    mask_uint8 = (seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay


def process_video_with_detection(
    input_path,
    output_path,
    conf_threshold=0.3,
    estimate_weight=False,
    weight_sample_interval=15,
    estimate_depth_fn=None,
    calc_weight_fn=None,
    scaling_factor_K=0.00016,  # corrected: 10x higher than image default
):
    """
    Reads a video, runs goat detection + green segmentation mask overlay on every
    frame, and optionally estimates weight every N sampled frames.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save annotated video.
        conf_threshold (float): Minimum confidence for detections.
        estimate_weight (bool): Whether to run weight estimation on sampled frames.
        weight_sample_interval (int): Run weight estimation every N frames.
        estimate_depth_fn (callable): estimate_depth_heatmap from depth_estimator.
        calc_weight_fn (callable): calculate_goat_volume_and_weight_proxy from depth_estimator.
        scaling_factor_K (float): Weight proxy scaling factor.

    Returns:
        tuple: (success: bool, weight_estimates: list[float])
    """
    if yolo_model is None:
        try:
            load_yolo_model()
        except Exception:
            logging.error("YOLO model not loaded, cannot process video.")
            return False, []

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {input_path}")
        return False, []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    weight_estimates = []
    frame_idx = 0
    current_weight_label = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            seg_mask_for_frame = None

            # --- YOLO: detection boxes + segmentation mask on every frame ---
            results = yolo_model(frame)
            for result in results:
                # Build combined binary segmentation mask
                if result.masks is not None and len(result.masks.data) > 0:
                    mask_data_np = result.masks.data.cpu().numpy()
                    combined_mask = np.zeros((height, width), dtype=np.uint8)
                    for mask_single in mask_data_np:
                        mask_resized = cv2.resize(
                            mask_single, (width, height),
                            interpolation=cv2.INTER_NEAREST
                        )
                        combined_mask = np.maximum(combined_mask, mask_resized)
                    seg_mask_for_frame = combined_mask

                # Draw bounding boxes
                boxes = result.boxes
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                        if conf < conf_threshold:
                            continue
                        cv2.rectangle(
                            frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame, f"goat {conf:.2f}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )
                break

            # --- Green mask overlay (same visual as image mode) ---
            if seg_mask_for_frame is not None:
                frame = _apply_green_mask_overlay(frame, seg_mask_for_frame)

            # --- Weight estimation on sampled frames ---
            if (
                estimate_weight
                and estimate_depth_fn is not None
                and calc_weight_fn is not None
                and frame_idx % weight_sample_interval == 0
                and seg_mask_for_frame is not None
            ):
                temp_frame_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        temp_frame_path = tmp.name
                        cv2.imwrite(temp_frame_path, frame)

                    _, raw_depth = estimate_depth_fn(temp_frame_path)

                    if raw_depth is not None:
                        _, weight_kg = calc_weight_fn(
                            raw_depth,
                            seg_mask_for_frame,
                            scaling_factor_K=scaling_factor_K,
                        )
                        if weight_kg is not None:
                            weight_estimates.append(weight_kg)
                            current_weight_label = f"Weight: {weight_kg:.1f} kg"
                            logging.info(
                                f"Frame {frame_idx}: weight proxy = {weight_kg:.2f} kg"
                            )

                except Exception as e:
                    logging.error(f"Weight estimation error on frame {frame_idx}: {e}")
                finally:
                    if temp_frame_path and os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)

            # --- Overlay weight label with dark background pill ---
            if current_weight_label:
                label_size, _ = cv2.getTextSize(
                    current_weight_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                )
                cv2.rectangle(frame, (8, 8), (label_size[0] + 18, 42), (0, 0, 0), -1)
                cv2.putText(
                    frame, current_weight_label,
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2
                )

            out.write(frame)
            frame_idx += 1

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
        return False, weight_estimates
    finally:
        cap.release()
        out.release()

    logging.info(
        f"Video saved to {output_path}. Weight samples: {len(weight_estimates)}"
    )
    return True, weight_estimates
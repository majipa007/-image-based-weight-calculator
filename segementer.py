from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            raise # Re-raise the exception to indicate failure

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
                    mask_resized = cv2.resize(mask_single, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, mask_resized)

                segmentation_mask = combined_mask
            else:
                logging.info(f"No masks detected for {image_path}.")

            segmented_img_pil = Image.fromarray(img_cv[..., ::-1]) # Convert BGR to RGB for PIL
            break # Process only the first result for simplicity

        logging.info("Segmentation completed.")
        return segmented_img_pil, segmentation_mask
    
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return None, None

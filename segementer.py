from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def segment_image(image_path):
    """
    Performs segmentation on an input image using a YOLOv8n-seg model.
    Args:
        image_path (str): Path to the input image.
    Returns:
        PIL.Image.Image: Annotated image with segmentation masks.
    """
    model = YOLO("model.pt")  # Load the custom segmentation model
    
    results = model(image_path)  # Predict on the image
    
    # Process results and draw masks
    for result in results:
        # Get the plotted image without any text or labels
        img_array = result.plot(labels=False, boxes=False) # Remove both labels and bounding boxes
        
        # Convert to OpenCV format for further processing
        img_cv = img_array.copy()
        
        segmentation_mask = None
        if result.masks is not None:
            # Combine all masks into a single binary mask
            # Assuming all masks are for goats, or we want a combined mask of all detected objects
            mask_data_np = result.masks.data.cpu().numpy()
            # Resize masks to original image size if they are not already
            original_h, original_w, _ = img_array.shape
            combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            for mask_single in mask_data_np:
                # Resize mask to original image dimensions
                mask_resized = cv2.resize(mask_single, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, mask_resized) # Combine masks

            segmentation_mask = combined_mask

            # The weight estimation part will be moved to app.py or a new utility function
            # For now, just return the image and the mask
        
        return Image.fromarray(img_cv[..., ::-1]), segmentation_mask  # Convert BGR to RGB for PIL and return mask
    
    return None, None # Return None for both if no results

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
        
        estimated_weights = []
        if result.masks is not None:
            for i, mask_data in enumerate(result.masks.data):
                # Convert mask to numpy array and calculate area
                mask_np = mask_data.cpu().numpy().astype(np.uint8)
                area_pixels = np.sum(mask_np) # Area in pixels

                # Placeholder for BMI-like calculation:
                # This is a simplified linear model. In a real-world scenario,
                # this would require calibration with actual goat weight data
                # and potentially more sophisticated image features.
                # Adjusted placeholder for BMI-like calculation:
                # The previous factor (0.001) resulted in weights that were too high.
                # This new factor (0.0002) is an adjustment to bring the estimated weights
                # closer to the 15-20 kg range for small goats.
                # This still requires proper calibration with actual goat weight data.
                weight_kg = area_pixels * 0.0002 
                estimated_weights.append(weight_kg)

                # Display the estimated weight on the image
                if result.boxes is not None and i < len(result.boxes):
                    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
                weight_label = f"Weight: {weight_kg:.2f} kg"
                
                # Calculate font scale and thickness relative to image size
                img_height, img_width, _ = img_cv.shape
                font_scale = min(img_height, img_width) * 0.0008 # Adjust this factor as needed
                font_thickness = max(1, int(min(img_height, img_width) * 0.0015)) # Adjust this factor as needed
                    
                # Put text above the bounding box (or near the segmented object)
                cv2.putText(img_cv, weight_label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return Image.fromarray(img_cv[..., ::-1])  # Convert BGR to RGB for PIL
    
    return None

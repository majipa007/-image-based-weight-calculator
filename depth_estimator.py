import cv2
import torch
import matplotlib.pyplot as plt
import logging
import time
import warnings
import numpy as np # Added for array operations
warnings.filterwarnings('ignore')

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global model and transforms to avoid reloading on every function call
midas = None
transform = None
device = None

def load_midas_model():
    """Loads the MiDaS model and transforms, and sets up the device."""
    global midas, transform, device
    if midas is None:
        logging.info("Loading MiDaS Model and Transforms...")
        try:
            model_type = "DPT_Large"
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            midas.to(device)
            midas.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform
            logging.info("MiDaS Model and Transforms Loaded Successfully.")
        except Exception as e:
            logging.error(f"Error while loading MiDaS model or transforms: {e}")
            midas = None # Ensure midas is None if loading fails
            transform = None
            device = None
            raise # Re-raise the exception to indicate failure

def estimate_depth_heatmap(image_path):
    """
    Estimates depth from an image and returns a matplotlib figure of the heatmap.
    Args:
        image_path (str): Path to the input image.
    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the depth heatmap.
                                  Returns None if an error occurs.
    """
    if midas is None or transform is None or device is None:
        try:
            load_midas_model()
        except Exception:
            return None # Return None if model loading fails

    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Could not read image from {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None

    try:
        logging.info(f"Depth prediction started for {image_path}...")
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        logging.info("Depth prediction completed.")

        # Create a matplotlib figure for the heatmap
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(output, cmap='magma')
        fig.colorbar(im, ax=ax, label='Depth Value')
        ax.set_title('Depth Estimation Heatmap')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.close(fig) # Close the figure to prevent it from being displayed immediately

        return fig, output
    except Exception as e:
        logging.error(f"Error during depth prediction: {e}")
        return None, None

def calculate_goat_volume_and_weight_proxy(raw_depth_map, segmentation_mask, scaling_factor_K=0.000016):
    """
    Calculates a volume proxy and a weight proxy for the goat.
    Args:
        raw_depth_map (np.array): The raw depth map from MiDaS.
        segmentation_mask (np.array): A binary mask of the goat (1 for goat, 0 for background).
        scaling_factor_K (float): Arbitrary scaling factor for weight calculation.
    Returns:
        tuple: (volume_proxy, weight_kg_proxy) or (None, None) if inputs are invalid.
    """
    if raw_depth_map is None or segmentation_mask is None:
        logging.error("Invalid inputs for volume and weight proxy calculation.")
        return None, None

    try:
        # Ensure mask and depth map have compatible shapes
        # The segmentation mask from segementer.py is already resized to original image dimensions
        # The raw_depth_map from estimate_depth_heatmap is also resized to original image dimensions
        # So, their shapes should match. If not, a warning is logged.
        if raw_depth_map.shape != segmentation_mask.shape:
            logging.warning(f"Depth map shape {raw_depth_map.shape} and segmentation mask shape {segmentation_mask.shape} do not match. This might lead to incorrect results.")
            # Attempt to resize mask to match depth map if necessary, though ideally they should match
            segmentation_mask = cv2.resize(segmentation_mask, 
                                           (raw_depth_map.shape[1], raw_depth_map.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)


        # Apply the segmentation mask to the depth map
        # Only consider depth values where the mask is active (goat pixels)
        masked_depth_values = raw_depth_map * (segmentation_mask > 0)

        # Calculate the volume proxy by summing the masked depth values
        volume_proxy = np.sum(masked_depth_values)

        # Calculate the weight proxy using the arbitrary scaling factor
        weight_kg_proxy = volume_proxy * scaling_factor_K

        logging.info(f"Volume Proxy: {volume_proxy:.2f}, Weight Proxy: {weight_kg_proxy:.2f} kg")
        return volume_proxy, weight_kg_proxy

    except Exception as e:
        logging.error(f"Error calculating volume and weight proxy: {e}")
        return None, None

# The standalone execution part is removed as this file will now primarily be imported.
# If you need to test it standalone, you can add a __main__ block:
# if __name__ == "__main__":
#     load_midas_model()
#     fig, raw_depth = estimate_depth_heatmap("test_images/1.jpg")
#     if fig:
#         plt.show()

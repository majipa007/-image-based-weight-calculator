# Goat Weight Detection and Depth Estimation

This project provides an image-based solution for detecting goats, estimating their depth, and calculating a proxy for their volume and weight. It leverages a custom-trained YOLOv8n-seg model for goat segmentation and a MiDaS model for monocular depth estimation. The application is built with Streamlit for an easy-to-use web interface.

## Features

*   **Goat Segmentation:** Accurately identifies and segments goats in uploaded images using a YOLOv8n-seg model.
*   **Depth Estimation:** Generates a depth heatmap for the uploaded image using a MiDaS model.
*   **Volume and Weight Proxy Estimation:** Calculates an approximate volume and weight for detected goats based on the segmented area and estimated depth.
*   **User-Friendly Interface:** A simple web application built with Streamlit for easy image uploads and result display.

## Project Structure

*   `app.py`: The main Streamlit application. It handles image uploads, resizing, temporary file management, orchestrates calls to the segmentation and depth estimation modules, and displays all results.
*   `segementer.py`: Contains the core logic for goat segmentation. It loads a cached YOLOv8n-seg model, performs predictions, and processes the results to generate a segmented image and a binary segmentation mask.
*   `depth_estimator.py`: Manages the depth estimation process. It loads a cached MiDaS model, estimates depth from the input image, generates a depth heatmap, and provides a function to calculate volume and weight proxies using the raw depth map and segmentation mask.
*   `model.pt`: The pre-trained YOLOv8n-seg model weights used for goat segmentation.
*   `requirements.txt`: Lists all the Python dependencies required to run the project, with pinned versions for reproducibility.

## Setup and Installation

To get this project up and running on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/majipa007/-image-based-weight-calculator.git
cd -image-based-weight-calculator
```

### 2. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Download the Model Weights

Ensure you have the `model.pt` file in the root directory of your project. This file contains the pre-trained weights for the YOLO segmentation model. If it's not present, the segmentation module will fail to load.

## How to Run the Application

Once you have completed the setup, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your web browser, usually at `http://localhost:8501`.

## Usage

1.  Open the Streamlit application in your browser.
2.  Click on "Choose an image..." to upload a JPG, JPEG, or PNG image containing one or more goats.
3.  The application will display the original image, then process it to perform segmentation and depth estimation.
4.  You will see the segmented image, a depth estimation heatmap, and the calculated volume and weight proxies.

## Technical Deep Dive

### `app.py` - The Streamlit Interface

This is the main entry point of the application. It sets up the Streamlit page configuration and title. Key functionalities include:
*   **Model Loading:** It uses `@st.cache_resource` to load both the MiDaS depth estimation model (via `depth_estimator.py`) and the YOLO segmentation model (via `segementer.py`) only once per session, optimizing performance.
*   **Image Handling:** It provides a file uploader for users to submit images. Uploaded images are opened with PIL and then resized to a `TARGET_MAX_SIDE` (e.g., 640 pixels) while preserving the aspect ratio, ensuring consistent input resolution for the models.
*   **Temporary Files:** Resized images are saved to a temporary file using Python's `tempfile` module, which ensures secure and automatic cleanup of these files after processing, even if errors occur.
*   **Orchestration:** It calls `segment_image` from `segementer.py` and `estimate_depth_heatmap` from `depth_estimator.py` to get the processed outputs.
*   **Result Display:** It uses Streamlit columns to display the original image, segmented image, depth heatmap, and the calculated volume and weight proxies with informative messages.

### `segementer.py` - Goat Segmentation Module

This module is responsible for identifying and segmenting goats within an image.
*   **YOLO Model Loading:** It uses a global variable `yolo_model` and a `load_yolo_model` function to load the `model.pt` (YOLOv8n-seg weights) only once. This model is specifically trained for goat segmentation.
*   **Segmentation Process:** The `segment_image` function takes an image path, feeds it to the loaded YOLO model for prediction.
*   **Mask Generation:** From the YOLO results, it extracts the segmentation masks. If multiple goats are detected, their masks are combined into a single binary mask. This mask is crucial for isolating the goat in the depth map.
*   **Output:** It returns the image with segmentation overlays and the raw binary segmentation mask as a NumPy array.

### `depth_estimator.py` - Depth Estimation Module

This module handles the monocular depth estimation and the proxy calculation for volume and weight.
*   **MiDaS Model Loading:** Similar to the YOLO model, it uses global variables (`midas`, `transform`, `device`) and a `load_midas_model` function to load the MiDaS `DPT_Large` model from `torch.hub.load` once. It automatically detects and uses a CUDA-enabled GPU if available.
*   **Depth Estimation:** The `estimate_depth_heatmap` function reads an image, transforms it for the MiDaS model, and performs inference to generate a raw depth map.
*   **Heatmap Generation:** It then uses `matplotlib` to create a visual depth heatmap from the raw depth map, which is displayed in the Streamlit app.
*   **Volume and Weight Proxy Calculation:** The `calculate_goat_volume_and_weight_proxy` function takes the raw depth map and the segmentation mask. It applies the mask to the depth map to isolate the depth values corresponding to the goat. The sum of these masked depth values forms the `volume_proxy`. This volume proxy is then multiplied by an arbitrary `scaling_factor_K` to derive a `weight_kg_proxy`.

## How Weight is Calculated (Step-by-Step)

The weight estimation in this application is a **proxy** and is derived through a series of image processing and model inference steps:

1.  **Image Upload and Resizing:** The user uploads an image. `app.py` resizes this image to a consistent maximum side (e.g., 640 pixels) to standardize input for the models.
2.  **Goat Segmentation:** The resized image is passed to `segementer.py`. The YOLOv8n-seg model identifies goats and generates a binary **segmentation mask** (a black and white image where white pixels represent the goat and black pixels are the background).
3.  **Depth Estimation:** Simultaneously, the resized image is passed to `depth_estimator.py`. The MiDaS model processes the image to produce a **raw depth map**, where pixel values represent the estimated distance of objects from the camera (darker areas are closer, lighter areas are farther).
4.  **Masking the Depth Map:** In `depth_estimator.py`, the segmentation mask is applied to the raw depth map. This effectively "cuts out" the goat from the depth map, leaving only the depth values that correspond to the goat's pixels.
5.  **Volume Proxy Calculation:** The sum of all depth values within the masked goat area is calculated. This sum serves as the **volume proxy**. It's an arbitrary unit because it lacks real-world camera calibration (focal length, sensor size, etc.).
6.  **Weight Proxy Calculation:** The `volume_proxy` is then multiplied by a predefined, arbitrary `scaling_factor_K`. This results in the **weight_kg_proxy**, which is an approximate weight in kilograms.

### Weight Estimation Disclaimer

The weight estimation provided by this model is a **proxy** and is based on a simplified relationship between the segmented area, estimated depth, and an arbitrary scaling factor. 

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository, create a new branch, and submit a pull request.


# Goat Weight Detection - Image Segmenter

This project provides an image-based solution for detecting goats and estimating their weight using segmentation. It leverages a custom-trained YOLOv8n-seg model to identify goats in images and then calculates an approximate weight based on the segmented area. The application is built with Streamlit for an easy-to-use web interface.

## Features

*   **Goat Segmentation:** Accurately identifies and segments goats in uploaded images.
*   **Weight Estimation:** Provides an estimated weight for each detected goat based on its segmented area.
*   **User-Friendly Interface:** A simple web application built with Streamlit for easy image uploads and result display.

## Project Structure

*   `app.py`: The main Streamlit application file that handles image uploads, calls the segmentation model, and displays results.
*   `segementer.py`: Contains the core segmentation logic, including loading the YOLO model and processing segmentation masks to estimate weight.
*   `model.pt`: The pre-trained YOLOv8n-seg model weights used for goat segmentation.
*   `requirements.txt`: Lists all the Python dependencies required to run the project.

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

Ensure you have the `model.pt` file in the root directory of your project. This file should be provided with the repository. If not, you would typically download it from a specified source or train your own model.

## How to Run the Application

Once you have completed the setup, you can run the Streamlit application:

```bash
streamlit run app.py
```

This command will open the application in your web browser, usually at `http://localhost:8501`.

## Usage

1.  Open the Streamlit application in your browser.
2.  Click on "Choose an image..." to upload a JPG, JPEG, or PNG image containing one or more goats.
3.  The application will display the original image, then process it to perform segmentation and weight estimation.
4.  Finally, it will show the segmented image with estimated weights overlaid on each detected goat.

## Weight Estimation Disclaimer

The weight estimation provided by this model is a **placeholder** and is based on a simplified linear relationship between the segmented area of a goat and its estimated weight. This model **requires proper calibration** with actual goat weight data and potentially more sophisticated image features for accurate results in a real-world scenario. It is intended for demonstration purposes and should not be used for critical applications without further development and validation.

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository, create a new branch, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).

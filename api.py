import base64
import io
import logging
import os
import tempfile

import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from depth_estimator import (
    calculate_goat_volume_and_weight_proxy,
    estimate_depth_heatmap,
    load_midas_model,
)
from segementer import load_yolo_model, segment_image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="Goat Analysis API", version="1.0.0")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_MAX_SIDE = 640


def pil_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a PNG data URL string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def fig_to_data_url(fig) -> str:
    """Convert a matplotlib figure to a PNG data URL string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.on_event("startup")
def startup_event() -> None:
    """Load models once when the API starts."""
    logger.info("Loading models on API startup...")
    try:
        load_midas_model()
        load_yolo_model()
        logger.info("Models loaded successfully.")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to load models on startup: %s", exc)


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image:
    - Resize to a standard size
    - Run segmentation and depth estimation
    - Compute volume and weight proxies
    - Return images and metrics in JSON.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400, content={"detail": "Uploaded file must be an image."}
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400, content={"detail": "Could not read image data."}
        )

    width, height = image.size
    if width > height:
        ratio = TARGET_MAX_SIDE / float(width)
        new_width = TARGET_MAX_SIDE
        new_height = int(height * ratio)
    else:
        ratio = TARGET_MAX_SIDE / float(height)
        new_height = TARGET_MAX_SIDE
        new_width = int(width * ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    temp_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_image_path = temp_file.name
            resized_image.save(temp_image_path)

        segmented_img, segmentation_mask = segment_image(temp_image_path)
        depth_fig, raw_depth_map = estimate_depth_heatmap(temp_image_path)

        volume_proxy = None
        weight_kg_proxy = None
        if segmentation_mask is not None and raw_depth_map is not None:
            (
                volume_proxy,
                weight_kg_proxy,
            ) = calculate_goat_volume_and_weight_proxy(raw_depth_map, segmentation_mask)

        segmented_image_data_url = (
            pil_to_data_url(segmented_img) if segmented_img is not None else None
        )
        depth_heatmap_data_url = (
            fig_to_data_url(depth_fig) if depth_fig is not None else None
        )

        if depth_fig is not None:
            plt.close(depth_fig)

        return {
            "original_width": width,
            "original_height": height,
            "resized_width": new_width,
            "resized_height": new_height,
            "segmented_image": segmented_image_data_url,
            "depth_heatmap": depth_heatmap_data_url,
            "volume_proxy": volume_proxy,
            "weight_kg_proxy": weight_kg_proxy,
            "note": (
                "Weight is an approximate proxy based on segmentation and depth; "
                "values are not calibrated to real-world measurements."
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during image analysis: %s", exc)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error."}
        )
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


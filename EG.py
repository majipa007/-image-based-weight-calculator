from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def segment_image(image_path):
    model = YOLO("model.pt")
    results = model(image_path)

    for result in results:
        img_array = result.plot(labels=False, boxes=False)
        img_cv = img_array.copy()
        segmentation_mask = None

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            h, w, _ = img_array.shape
            combined = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                combined = np.maximum(combined, m_resized)
            segmentation_mask = combined

            return Image.fromarray(img_cv[..., ::-1]), segmentation_mask

    return None, None

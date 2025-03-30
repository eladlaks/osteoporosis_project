import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL image to numpy array (OpenCV format)
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img_clahe = clahe.apply(
            cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        )  # Applying CLAHE on grayscale image

        # Convert back to PIL image
        img_clahe = Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB))
        return img_clahe

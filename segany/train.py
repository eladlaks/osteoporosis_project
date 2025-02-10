import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# Load the model (choose from 'vit_h', 'vit_l', or 'vit_b')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "segany/sam_vit_h_4b8939.pth"  # Download from Meta's SAM GitHub repo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)

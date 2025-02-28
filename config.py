import os
import torch

# Data configuration
DATA_DIR = os.path.join(os.getcwd(), "data", "images")  # Directory containing images (sub-folders for each class)
DUPLICATE_THRESHOLD = 0.99  # (Not used directly here but can be adapted for more advanced duplicate checking)

# Preprocessing configuration
YOLO_WEIGHTS_PATH = os.path.join(os.getcwd(), "pretrained", "yolo_weights.pt")  # Update with your YOLO weights path

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 3
ALEX_FREEZE_FEATURES = True
USE_TRANSFORM_AUGMENTATION_IN_TRAINING = False
USE_UNKNOW_CODE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Augmentation configuration (example)
AUGMENTATION_PROB = 0.5


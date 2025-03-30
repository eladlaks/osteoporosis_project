import os
import torch

# Data configuration
DATA_DIR = os.path.join(
    os.getcwd(), "data"
)  # Directory containing images (sub-folders for each class)
DUPLICATE_THRESHOLD = 0.99  # (Not used directly here but can be adapted for more advanced duplicate checking)

# Preprocessing configuration
YOLO_WEIGHTS_PATH = os.path.join(
    os.getcwd(), "pretrained", "yolo_weights.pt"
)  # Update with your YOLO weights path
YOLO_WEIGHTS_PATH = os.path.join(
    os.getcwd(), "pretrained", "yolo_weights.pt"
)  # Update with your YOLO weights path

# Training configuration
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 0.03
NUM_CLASSES = 3
ALEX_FREEZE_FEATURES = True
USE_TRANSFORM_AUGMENTATION_IN_TRAINING = False
USE_CLAHE = True
USE_UNKNOW_CODE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_OSTEOPENIA = False
SKIP_DUP_DATA = False
# Augmentation configuration (example)
AUGMENTATION_PROB = 0.5

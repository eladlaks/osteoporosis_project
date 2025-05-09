import os
import torch

# Data configuration
DATA_DIR = os.path.join(os.getcwd(), "data//cropped_data")
TEST_DATA_DIR = os.path.join(os.getcwd(), "data//test_cropped_data")
DUPLICATE_THRESHOLD = 0.99  # (Not used directly here but can be adapted for more advanced duplicate checking)

YOLO_WEIGHTS_PATH = os.path.join(
    os.getcwd(), "pretrained", "yolo_weights.pt"
)  # Update with your YOLO weights path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.000001
ALEX_FREEZE_FEATURES = True
USE_TRANSFORM_AUGMENTATION_IN_TRAINING = True
USE_CLAHE = False
MODEL_NAME = "ResNet50"
USE_OSTEOPENIA = False
SKIP_DUP_DATA = False
# Augmentation configuration (example)
TRAIN_WEIGHTED_RANDOM_SAMPLER = True
NUM_WORKERS = 1
USE_METABOLIC_FOR_TEST = True
USE_SCHEDULER = False

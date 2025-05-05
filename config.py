import os
import torch

# Data configuration
DATA_DIR = os.path.join(os.getcwd(), "data//cropped_data")
TEST_DATA_DIR = os.path.join(os.getcwd(), "data//test_cropped_data")
DUPLICATE_THRESHOLD = 0.99  # (Not used directly here but can be adapted for more advanced duplicate checking)

YOLO_WEIGHTS_PATH = os.path.join(
    os.getcwd(), "pretrained", "yolo_weights.pt"
)  # Update with your YOLO weights path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 0.000001
NUM_CLASSES = 3
ALEX_FREEZE_FEATURES = True
USE_TRANSFORM_AUGMENTATION_IN_TRAINING = True
USE_CLAHE = False
MODEL_NAME = "ResNet50"
USE_OSTEOPENIA = True
SKIP_DUP_DATA = False

# Augmentation and sampling
TRAIN_WEIGHTED_RANDOM_SAMPLER = True
NUM_WORKERS = 4
USE_METABOLIC_FOR_TEST = True
USE_SCHEDULER = False

# Merge predictions into metabolic Excel file at test time
MERGE_RESULTS_TO_METABOLIC_DF = True
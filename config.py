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
NUM_EPOCHS = 2
LEARNING_RATE = 0.000001
ALEX_FREEZE_FEATURES = True
USE_TRANSFORM_AUGMENTATION_IN_TRAINING = False
USE_CLAHE = False
MODEL_NAME = "ResNet50"
USE_OSTEOPENIA = True  # Set to False if you want to exclude Osteopenia class from training
SKIP_DUP_DATA = False
# Augmentation configuration (example)
TRAIN_WEIGHTED_RANDOM_SAMPLER = True
NUM_WORKERS = 1
USE_METABOLIC_FOR_TEST = True
USE_SCHEDULER = False
# Custom Techniques
USE_LABEL_SMOOTHING = True
USE_CONFIDENCE_WEIGHTED_LOSS = True
USE_HARD_SAMPLING = True
CONFIDENCE_THRESHOLD = 0.75
CONFIDENCE_PENALTY_WEIGHT = 2.0
LABEL_SMOOTHING_EPSILON = 0.1
# ─── Ensemble defaults ─────────────────────────────────────────────
ENSEMBLE_TYPE    = "none"           # none | soft | weighted | stacking
CKPT_LIST        = []               # will be filled via CLI / sweep
ARCH_LIST        = []               # e.g. ["resnet50", "vgg19", "vit"]
ENSEMBLE_WEIGHTS = []               # only for weighted voting
META_CLF_PATH    = ""               # only for stacking

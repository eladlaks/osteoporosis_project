from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
import wandb
from models.vgg19_model import get_vgg19_model
from models.vit_model import get_vit_model
from models.alexnet_model import get_alexnet_model
from models.gideon_alex_net import get_gideon_alexnet_model
from models.resnet_model import get_resnet_model
from models.resnet_model import get_timm_model
from models.dino_model import get_dinov2_model
from torchvision import transforms
from preprocessing.clahe import CLAHETransform
from utils.logger import init_wandb
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from losses.label_smoothing import LabelSmoothingCrossEntropy
from losses.confidence_weighted_loss import ConfidenceWeightedCrossEntropy
from losses.combined_loss import CombinedLabelSmoothingConfidenceWeightedLoss

from utils.hard_sampling import get_low_confidence_samples
from utils.saver import save_test_outputs
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from dataset_handler.dataset import ImageDataset
from preprocessing.clahe import CLAHETransform
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np

# Your ensemble_test_predictions function goes here (already provided)

# Example model class — replace with your actual one
from torchvision.models import resnet50
from torch.nn import Linear


def ensemble_test_predictions(models, test_dataset):
    loader = DataLoader(test_dataset, batch_size=32)
    all_probs = []
    true_labels = []
    filenames = []

    for images, labels, paths in loader:
        images = images.to("cuda")
        batch_probs = []

        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs.cpu())

        avg_probs = torch.stack(batch_probs).mean(dim=0)
        all_probs.append(avg_probs)
        true_labels.extend(labels.numpy())
        filenames.extend(paths)

    all_probs = torch.cat(all_probs).numpy()
    predicted_labels = np.argmax(all_probs, axis=1)
    return filenames, true_labels, predicted_labels, all_probs


def create_model(num_classes):
    model = resnet50(pretrained=False)
    model.fc = Linear(model.fc.in_features, num_classes)
    return model


def load_models_from_folder(folder_path, num_classes, device="cuda"):
    model_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pth")
    ]
    models = []

    for path in model_paths:
        model_class_name = os.path.split(path)[-1].split("_")[0]
        if model_class_name == "ResNet50":
            model = get_resnet_model(path)
        else:
            model = get_timm_model(path, model_class_name)

        # model = create_model(num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        models.append(model)

    return models


def evaluate_predictions(true_labels, predicted_labels, all_probs, class_names):
    # Accuracy
    acc = accuracy_score(true_labels, predicted_labels)

    # F1 Score (macro for multi-class)
    f1 = f1_score(true_labels, predicted_labels, average="macro")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # AUC (One-vs-Rest strategy)
    try:
        auc = roc_auc_score(true_labels, all_probs, multi_class="ovr")
    except ValueError:
        auc = None  # Sometimes fails if only one class in y_true

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    if auc is not None:
        print(f"AUC (OvR): {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(true_labels, predicted_labels, target_names=class_names)
    )

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return auc, f1, acc, cm


def ensemble_models_from_path(folder_path, class_names, test_dataset):
    num_classes = len(class_names)
    models = load_models_from_folder(folder_path, num_classes)
    filenames, true_labels, predicted_labels, all_probs = ensemble_test_predictions(
        models, test_dataset
    )
    class_names = ["normal", "osteopenia", "osteoporosis"]
    auc, f1, acc, cm = evaluate_predictions(
        true_labels, predicted_labels, all_probs, class_names
    )

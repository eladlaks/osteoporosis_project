import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader
import wandb
from dataset_handler.dataset import ImageDataset
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

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from dataset_handler.dataset import ImageDataset
from preprocessing.clahe import CLAHETransform
import tqdm
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs, scheduler=None, use_wandb=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm.tqdm(dataloader, desc=f"Epoch [{epoch+1}/{total_epochs}]", leave=False)
    
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    if use_wandb:
        wandb.log({
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_acc,
            "Epoch": epoch + 1
        })

    return epoch_loss, epoch_acc


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def run_fold(fold, model, optimizer, criterion, train_idx, val_idx, test_loader, full_dataset,config):
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE)

    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    no_improve_epochs = 0

    for epoch in range(config.NUM_EPOCHS):
        
        train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE, epoch, config.NUM_EPOCHS)
        val_acc = evaluate(model, val_loader, config.DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_loader, config.DEVICE)
    print(f"Fold {fold + 1} — Val Accuracy: {best_val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    wandb.log({"fold_test_accuracy": test_acc})

    return model


def ensemble_test_predictions(models, test_dataset):
    loader = DataLoader(test_dataset, batch_size=wandb.config.BATCH_SIZE)
    all_probs = []
    true_labels = []
    filenames = []

    for images, labels, paths in loader:
        images = images.to(wandb.config.DEVICE)
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


def run_kfold_cross_validation(model_class, optimizer_class, criterion, args,config):

    size = (518, 518) if config.MODEL_NAME == "DINOv2" else (512, 512)
    prepare_to_network_transforms = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    augmentation_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]

    all_transformation = []
    train_transformations = []
    if config.USE_TRANSFORM_AUGMENTATION_IN_TRAINING:
        train_transformations += augmentation_transform
    if config.USE_CLAHE:
        all_transformation += [
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
        ]

    all_transformation += prepare_to_network_transforms
    train_transformations += all_transformation
    eval_transform = transforms.Compose(all_transformation)
    train_transform = transforms.Compose(train_transformations)

    full_dataset = ImageDataset(config.DATA_DIR, transform=train_transform)
    test_dataset = ImageDataset(config.TEST_DATA_DIR, transform=eval_transform)
    config.NUM_CLASSES = len(set(full_dataset.labels))
    device = config.DEVICE
    batch_size = config.BATCH_SIZE
    kfold = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)
    models = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        model = model_class.to(device)
        optimizer = optimizer_class

        trained_model = run_fold(
            fold, model, optimizer, criterion,
            train_idx, val_idx,
            DataLoader(test_dataset, batch_size=batch_size),
            full_dataset,
            config
        )
        models.append(trained_model)

    # Ensemble prediction and export
    filenames, y_true, y_pred, y_probs = ensemble_test_predictions(models, test_dataset)
    class_names = [str(i) for i in range(config.NUM_CLASSES)]
    df = pd.DataFrame({
        "filename": filenames,
        "true_label": y_true,
        "predicted_label": y_pred,
    })
    for i, cls in enumerate(class_names):
        df[f"prob_class_{cls}"] = y_probs[:, i]
    df.to_csv("test_predictions.csv", index=False)
    print("Saved predictions to test_predictions.csv")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Ensemble)")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to confusion_matrix.png")

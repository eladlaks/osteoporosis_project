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


WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler=None,
):
    model.to(wandb.config.DEVICE)
    best_val_loss = float("inf")  # Initialize best validation loss
    best_model_path = os.path.join("saved_models", f"{model_name}_best.pth")
    os.makedirs("saved_models", exist_ok=True)
    best_model = model
    for epoch in range(wandb.config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(wandb.config.DEVICE)
            labels = labels.to(wandb.config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(
            f"[{model_name}] Epoch {epoch+1}/{wandb.config.NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}"
        )
        wandb.log({f"train_loss": epoch_loss, "epoch": epoch + 1})

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(wandb.config.DEVICE)
                labels = labels.to(wandb.config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        print(
            f"[{model_name}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
        wandb.log(
            {
                f"val_loss": avg_val_loss,
                f"val_acc": val_accuracy,
                "epoch": epoch + 1,
            }
        )

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            print(f"Best model with validation loss: {best_val_loss:.4f}")

        # Step the scheduler with validation loss
        if scheduler:
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

    # Save the last model weights after training
    model_save_path = os.path.join("saved_models", f"{model_name}.pth")
    torch.save(model.state_dict(), model_save_path)
    torch.save(best_model.state_dict(), best_model_path)

    print(f"Saved last {model_name} model to {model_save_path}")
    print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # Final evaluation on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(wandb.config.DEVICE)
            labels = labels.to(wandb.config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            # else:
            #     probs = torch.softmax(outputs, dim=1)
            #     predicted = torch.tensor(
            #         [0 if p[0] > 0.6 else 1 if p[1] > 0.6 else 2 for p in probs]
            #     ).to(wandb.config.DEVICE)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total

    # Convert to numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # Compute evaluation metrics
    test_f1 = f1_score(all_labels_np, all_preds_np, average="macro")
    test_precision = precision_score(all_labels_np, all_preds_np, average="macro")
    test_recall = recall_score(all_labels_np, all_preds_np, average="macro")

    try:
        all_labels_one_hot = np.eye(wandb.config.NUM_CLASSES)[all_labels_np]
        test_auc = roc_auc_score(
            all_labels_one_hot, all_probs_np, average="macro", multi_class="ovr"
        )
    except Exception as e:
        print(f"AUC computation failed: {e}")
        test_auc = None

    if test_auc is not None:
        auc_str = f"{test_auc:.4f}"
    else:
        auc_str = "N/A"

    print(
        f"[{model_name}] Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
        f"F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, AUC: {auc_str}"
    )

    wandb.log(
        {
            f"test_loss": avg_test_loss,
            f"test_acc": test_accuracy,
            f"test_f1": test_f1,
            f"test_precision": test_precision,
            f"test_recall": test_recall,
            f"test_auc": test_auc if test_auc is not None else 0.0,
        }
    )
    # Confusion Matrix
    if wandb.config.NUM_CLASSES == 2:
        cm = confusion_matrix(all_labels_np, all_preds_np)
        class_names = ["Normal", "Osteoporosis"]
    else:
        cm = confusion_matrix(all_labels_np, all_preds_np)
        class_names = ["Normal", "Osteopenia", "Osteoporosis"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    wandb.log({f"{model_name}_confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Classification Report + Per-Class Metrics
    report = classification_report(
        all_labels_np, all_preds_np, output_dict=True, zero_division=0
    )
    wandb.log({f"{model_name}_classification_report": report})

    # Log per-class metrics (optional)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                wandb.log({f"{model_name}_{label}_{metric_name}": value})


def run_training(args):
    # Initialize wandb for this run

    wandb.login(key=WANDB_API_KEY)
    init_wandb(project_name="image_classification_project", args=args)
    size = (518, 518) if wandb.config.MODEL_NAME == "DINOv2" else (512, 512)
    prepare_to_network_transforms = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    augmentation_transform = [
        transforms.RandomRotation(degrees=10),  # Small random rotation
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small shifts
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust contrast
    ]

    all_transformation = []
    train_transformations = []
    if wandb.config.USE_TRANSFORM_AUGMENTATION_IN_TRAINING:
        train_transformations += augmentation_transform
    if wandb.config.USE_CLAHE:
        all_transformation += (
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        )  # Apply CLAHE with custom parameters

    all_transformation += prepare_to_network_transforms
    train_transformations += all_transformation
    eval_transform = transforms.Compose(all_transformation)

    train_transform = transforms.Compose(train_transformations)

    # Load the full dataset
    full_dataset = ImageDataset(wandb.config.DATA_DIR)
    wandb.config.NUM_CLASSES = len(set(full_dataset.labels))
    total_size = len(full_dataset)
    if wandb.config.USE_METABOLIC_FOR_TEST:
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        test_dataset = ImageDataset(wandb.config.TEST_DATA_DIR)
        test_dataset.transform = eval_transform

    else:
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        # Randomly split the dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        test_dataset.dataset.transform = eval_transform

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform

    # Check if we should use weighted sampler
    if wandb.config.TRAIN_WEIGHTED_RANDOM_SAMPLER:
        # Compute class distribution for the training dataset
        labels = [
            train_dataset.dataset[i][1] for i in train_dataset.indices
        ]  # Extract labels
        class_counts = Counter(labels)
        num_samples = max(class_counts.values()) * len(
            class_counts
        )  # Total balanced samples

        # Compute class weights (inverse of frequencies)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        weights = [class_weights[label] for label in labels]

        # Define sampler for balanced training
        train_sampler = WeightedRandomSampler(weights, num_samples, replacement=True)
    else:
        train_sampler = None

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=wandb.config.BATCH_SIZE,
        sampler=(
            train_sampler if wandb.config.TRAIN_WEIGHTED_RANDOM_SAMPLER else None
        ),  # Apply sampler if enabled
        shuffle=not wandb.config.TRAIN_WEIGHTED_RANDOM_SAMPLER,  # Shuffle only if not using sampler
        num_workers=wandb.config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=wandb.config.BATCH_SIZE,
        shuffle=False,
        num_workers=wandb.config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=wandb.config.BATCH_SIZE,
        shuffle=False,
        num_workers=wandb.config.NUM_WORKERS,
    )
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # List of models to train
    model_name = wandb.config.MODEL_NAME
    if model_name == "VGG19":
        model_func = get_vgg19_model
    elif model_name == "ViT":
        model_func = get_vit_model
    elif model_name == "AlexNet":
        model_func = get_alexnet_model
    elif model_name == "Gideon_Alexnet":
        model_func = get_gideon_alexnet_model
    elif model_name == "ResNet50":
        model_func = get_resnet_model
    elif model_name == "DINOv2":
        model_func = get_dinov2_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    print(f"Training {model_name} model...")
    model = model_func()
    optimizer = optim.Adam(
        model.parameters(), lr=wandb.config.LEARNING_RATE, weight_decay=1e-5
    )
    if wandb.config.USE_SCHEDULER:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.5, verbose=True
        )
    else:
        scheduler = None
    train_model(
        model,
        model_name,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
    )

    wandb.finish()

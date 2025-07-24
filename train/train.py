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
    eval_transform=None,
    train_dataset=None,  # Pass the train_dataset for low-confidence sampling
):
    model.to(wandb.config.DEVICE)
    best_val_loss = float("inf")  # Initialize best validation loss
    best_model_path = os.path.join("saved_models", f"{model_name}_best.pth")
    os.makedirs("saved_models", exist_ok=True)
    best_model = model
    for epoch in range(wandb.config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
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
            for images, labels, _ in val_loader:
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
    print(
        f"Best model saved with validation loss: {best_val_loss:.4f}"
    )  # Final evaluation on test set
    use_best_model_gideon(best_model,model_name,best_model_path,test_loader,criterion)

    # Save low-confidence samples for hard sampling===
    if wandb.config.USE_HARD_SAMPLING:
        # Only use training data for hard sampling to avoid leakage
        train_eval_loader = DataLoader(
            train_dataset,  # not full_dataset!
            batch_size=wandb.config.BATCH_SIZE,
            shuffle=False
        )

        low_conf_paths = get_low_confidence_samples(
            model,
            train_eval_loader,
            threshold=wandb.config.CONFIDENCE_THRESHOLD,
            device=wandb.config.DEVICE,
        )

        with open("low_conf_samples.txt", "w") as f:
            for path in low_conf_paths:
                f.write(path + "\n")

        print(
            f"{len(low_conf_paths)} low-confidence samples saved to low_conf_samples.txt"
        )


def run_training(args):
    # Initialize wandb for this run

    wandb.login(key=WANDB_API_KEY)
    init_wandb(project_name="final_project", args=args)
    size = (518, 518) if wandb.config.MODEL_NAME == "DINOv2" else (512, 512)
    prepare_to_network_transforms = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    augmentation_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(degrees=10),  # Small random rotation
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small shifts
        transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2, 
                           saturation=0.2, 
                           hue=0.1),  # Adjust contrast
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
    if wandb.config.USE_TEST_DATA_DIR:
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
    # Logic to select the appropriate loss function
    if wandb.config.USE_LABEL_SMOOTHING and wandb.config.USE_CONFIDENCE_WEIGHTED_LOSS:
        criterion = CombinedLabelSmoothingConfidenceWeightedLoss(
            epsilon=wandb.config.LABEL_SMOOTHING_EPSILON,
            threshold=wandb.config.CONFIDENCE_THRESHOLD,
            penalty_factor=wandb.config.CONFIDENCE_PENALTY_WEIGHT,
            reduction="mean",
        )
    elif wandb.config.USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(
            epsilon=wandb.config.LABEL_SMOOTHING_EPSILON, reduction="mean"
        )
    elif wandb.config.USE_CONFIDENCE_WEIGHTED_LOSS:
        criterion = ConfidenceWeightedCrossEntropy(
            threshold=wandb.config.CONFIDENCE_THRESHOLD,
            penalty_factor=wandb.config.CONFIDENCE_PENALTY_WEIGHT,
            reduction="mean",
        )
    else:
        criterion = nn.CrossEntropyLoss()

    wandb.log({"loss_type": criterion.__class__.__name__})

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
    elif model_name == "resnet34" or model_name == "resnet50" or model_name == "densenet121" or model_name == "efficientnet_b0":
        model_func = get_timm_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    print(f"Training {model_name} model...")
    model = model_func()
    optimizer = optim.Adam(
        model.parameters(), lr=wandb.config.LEARNING_RATE, weight_decay=1e-4
    )
    if wandb.config.USE_SCHEDULER:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.2, verbose=True
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
        eval_transform=eval_transform,
        train_dataset=train_dataset,  # Pass the train_dataset for low-confidence sampling
    )
    # ==== Optional Fine-Tuning on Low Confidence Samples ====
    if wandb.config.USE_HARD_SAMPLING:
        from dataset_handler.filtered_dataset import FilteredImageDataset

        # Load low-confidence paths
        with open("low_conf_samples.txt", "r") as f:
            selected_paths = set(line.strip() for line in f.readlines())

        # Create dataset and loader
        hard_dataset = FilteredImageDataset(
            root_dir=wandb.config.DATA_DIR,
            selected_paths_set=selected_paths,
            transform=train_transform,
        )
        hard_loader = DataLoader(
            hard_dataset,
            batch_size=wandb.config.BATCH_SIZE,
            shuffle=True,
            num_workers=wandb.config.NUM_WORKERS,
        )
        

        # Reinitialize optimizer with difernatial learning rate
        fine_tune_lr = wandb.config.LEARNING_RATE * wandb.config.FINE_TUNE_LR_MULTIPLIER
        optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr, weight_decay=1e-5)

        print("Starting fine-tuning on low-confidence samples...")
        train_model(
            model,
            model_name + "_hard_finetune",
            hard_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            scheduler,
            eval_transform=eval_transform,
            train_dataset=train_dataset,  # Pass the train_dataset for low-confidence sampling
        )
    wandb.finish()


def use_best_model(best_model,model_name,best_model_path,test_loader,criterion):

    artifact = wandb.Artifact(f"best_model_{model_name}", type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)
    # Log the best model weights to wandb

    best_model.load_state_dict(torch.load(best_model_path, weights_only=False))
    best_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_images_path = []
    all_image_legs = []

    with torch.no_grad():
        for images, labels, images_path in test_loader:
            images = images.to(wandb.config.DEVICE)
            labels = labels.to(wandb.config.DEVICE)
            outputs = best_model(images)
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
            all_images_path.extend(
                [
                    os.path.splitext(os.path.basename(p))[0].split("_")[0]
                    for p in images_path
                ]
            )
            all_image_legs.extend(
                [
                    os.path.splitext(os.path.basename(p))[0].split("_")[1]
                    for p in images_path
                ]
            )

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
    wandb.log(
        {f"{model_name}_classification_report": report}
    )  # Log per-class metrics (optional)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                wandb.log({f"{model_name}_{label}_{metric_name}": value})

    # Process patient details CSV and merge with predictions if using metabolic test data
    if wandb.config.USE_TEST_DATA_DIR:
        try:
            # Create predictions DataFrame
            df_pred = pd.DataFrame(
                {
                    "labels": all_labels,
                    "preds": all_preds,
                    "probs": all_probs,
                    "path": all_images_path,
                    "leg_tag": all_image_legs,
                }
            )

            # Load patient details CSV (try both Excel and CSV formats)
            patient_details_path = "data/new_data/patient_details.csv"
            if os.path.exists(patient_details_path):
                df_patient = pd.read_csv(patient_details_path)
            else:
                print("Warning: Patient details file not found")
                df_patient = None

            if df_patient is not None:
                # Ensure the DataFrame has the necessary structure for probabilities
                if "probs_class_0" not in df_patient.columns:
                    df_patient["probs_class_0"] = None
                    df_patient["probs_class_1"] = None
                    df_patient["probs_class_2"] = None

                # Merge predictions with patient details
                df_merged = df_patient.reset_index().merge(
                    df_pred, left_on="Patient Id", right_on="path"
                )

                # Extract individual probability classes
                df_merged[["probs_class_0", "probs_class_1", "probs_class_2"]] = (
                    pd.DataFrame(df_merged["probs"].tolist(), index=df_merged.index)
                )

                # Save the merged DataFrame
                output_csv_path = f"patient_details_with_probs_output_{model_name}.csv"
                df_merged.to_csv(output_csv_path, index=False)

                # Log the CSV file to wandb
                artifact = wandb.Artifact(f"test_metabolic_{model_name}", type="csv")
                artifact.add_file(output_csv_path)
                wandb.log_artifact(artifact)

                print(f"Patient details with predictions saved to: {output_csv_path}")
                print(f"CSV file logged to wandb")

                # Log some summary statistics
                wandb.log(
                    {
                        f"{model_name}_total_patients": len(df_merged),
                        f"{model_name}_avg_prob_normal": df_merged[
                            "probs_class_0"
                        ].mean(),
                        f"{model_name}_avg_prob_osteopenia": df_merged[
                            "probs_class_1"
                        ].mean(),
                        f"{model_name}_avg_prob_osteoporosis": df_merged[
                            "probs_class_2"
                        ].mean(),
                    }
                )

        except Exception as e:
            print(f"Error processing patient details: {e}")



def use_best_model_gideon(best_model,model_name,best_model_path,test_loader,criterion):

    artifact = wandb.Artifact(f"best_model_{model_name}", type="model")
    artifact.add_file(best_model_path)
    wandb.log_artifact(artifact)
    # Log the best model weights to wandb

    best_model.load_state_dict(torch.load(best_model_path, weights_only=False))
    best_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_images_path = []
    all_image_legs = []
    
# ----- save *single* logits/labels file -----
    run_tag = f"{model_name}_best"                  # <-- שם קבוע
    # Validation step
    best_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    test_logits_batches, test_label_batches, test_path_batches = [], [], []

    # remove previous version if it exists
    from pathlib import Path
    pt_prev  = Path("saved_models") / f"{run_tag}_val_logits.pt"
    csv_prev = pt_prev.with_suffix(".csv")
    if pt_prev.exists():  pt_prev.unlink()
    if csv_prev.exists(): csv_prev.unlink()

    

    with torch.no_grad():
        for images, labels, images_path in test_loader:
            images = images.to(wandb.config.DEVICE)
            labels = labels.to(wandb.config.DEVICE)
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            test_logits_batches.append(outputs.cpu())
            test_label_batches.append(labels.cpu())
            test_path_batches.extend(images_path)
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
            all_images_path.extend(
                [
                    os.path.splitext(os.path.basename(p))[0].split("_")[0]
                    for p in images_path
                ]
            )
            all_image_legs.extend(
                [
                    os.path.splitext(os.path.basename(p))[0].split("_")[1]
                    for p in images_path
                ]
            )


    pt_file = save_test_outputs(
        run_tag=run_tag,
        logits=torch.cat(test_logits_batches),
        labels=torch.cat(test_label_batches),
        img_paths=test_path_batches
    )

    # ----- W&B artifact (auto-versioned) -----
    artifact = wandb.Artifact(
        f"{model_name}_test_outputs", type="model-eval",
        metadata={"source": "best"}                 # optional meta
    )
    artifact.add_file(best_model_path)              # weights
    artifact.add_file(pt_file)                      # logits+labels
    csv_file = pt_file.with_suffix(".csv")
    if csv_file.exists():
        artifact.add_file(csv_file)                 # CSV view
    wandb.log_artifact(artifact)

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
    wandb.log(
        {f"{model_name}_classification_report": report}
    )  # Log per-class metrics (optional)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                wandb.log({f"{model_name}_{label}_{metric_name}": value})

    # Process patient details CSV and merge with predictions if using metabolic test data
    if wandb.config.USE_METABOLIC_FOR_TEST:
        try:
            # Create predictions DataFrame
            df_pred = pd.DataFrame(
                {
                    "labels": all_labels,
                    "preds": all_preds,
                    "probs": all_probs,
                    "path": all_images_path,
                    "leg_tag": all_image_legs,
                }
            )

            # Load patient details CSV (try both Excel and CSV formats)
            patient_details_path = "data/new_data/patient_details.csv"
            if os.path.exists(patient_details_path):
                df_patient = pd.read_csv(patient_details_path)
            else:
                print("Warning: Patient details file not found")
                df_patient = None

            if df_patient is not None:
                # Ensure the DataFrame has the necessary structure for probabilities
                if "probs_class_0" not in df_patient.columns:
                    df_patient["probs_class_0"] = None
                    df_patient["probs_class_1"] = None
                    df_patient["probs_class_2"] = None

                # Merge predictions with patient details
                df_merged = df_patient.reset_index().merge(
                    df_pred, left_on="Patient Id", right_on="path"
                )

                # Extract individual probability classes
                df_merged[["probs_class_0", "probs_class_1", "probs_class_2"]] = (
                    pd.DataFrame(df_merged["probs"].tolist(), index=df_merged.index)
                )

                # Save the merged DataFrame
                output_csv_path = f"patient_details_with_probs_output_{model_name}.csv"
                df_merged.to_csv(output_csv_path, index=False)

                # Log the CSV file to wandb
                artifact = wandb.Artifact(f"test_metabolic_{model_name}", type="csv")
                artifact.add_file(output_csv_path)
                wandb.log_artifact(artifact)

                print(f"Patient details with predictions saved to: {output_csv_path}")
                print(f"CSV file logged to wandb")

                # Log some summary statistics
                wandb.log(
                    {
                        f"{model_name}_total_patients": len(df_merged),
                        f"{model_name}_avg_prob_normal": df_merged[
                            "probs_class_0"
                        ].mean(),
                        f"{model_name}_avg_prob_osteopenia": df_merged[
                            "probs_class_1"
                        ].mean(),
                        f"{model_name}_avg_prob_osteoporosis": df_merged[
                            "probs_class_2"
                        ].mean(),
                    }
                )

        except Exception as e:
            print(f"Error processing patient details: {e}")
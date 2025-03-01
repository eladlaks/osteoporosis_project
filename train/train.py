import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from config import DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE,USE_TRANSFORM_AUGMENTATION_IN_TRAINING,USE_UNKNOW_CODE
from dataset_handler.dataset import ImageDataset
from models.vgg19_model import get_vgg19_model
from models.vit_model import get_vit_model
from models.alexnet_model import get_alexnet_model
from models.resnet_model import get_resnet_model
from torchvision import transforms

def train_model(model, model_name, dataloader, criterion, optimizer):
    model.to(DEVICE)
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"[{model_name}] Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
        
        # Log metrics to wandb
        wandb.log({f"{model_name}_loss": epoch_loss, "epoch": epoch+1})
    
    # Save model weights
    model_save_path = os.path.join("saved_models", f"{model_name}.pth")
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved {model_name} model to {model_save_path}")

def run_training():
    # Initialize wandb for this run
    wandb.init(project="image_classification_project", reinit=True)
    
    # Define transformations (resize, tensor conversion, normalization)
    if USE_TRANSFORM_AUGMENTATION_IN_TRAINING:
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomRotation(degrees=10),  # Small random rotation
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small shifts
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for medical images
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Load the full dataset
    full_dataset = ImageDataset(DATA_DIR)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Randomly split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform
    if USE_UNKNOW_CODE:
        # **Compute Class Distribution for Weighted Sampling**
        class_counts = torch.bincount(torch.tensor([label for _, label in full_dataset.samples]))

        # **Ensure All Classes Exist (Avoid Zero-Division Errors)**
        total_samples = sum(class_counts).item()
        class_weights = total_samples / class_counts  # Inverse frequency weighting
        sample_weights = torch.tensor([class_weights[label] for _, label in full_dataset.samples])

        # **Apply WeightedRandomSampler ONLY to Training Set**
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights[train_dataset.indices], num_samples=len(train_dataset), replacement=True
        )

        # **DataLoaders (Batch Size & Sampler Only for Training)**
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # List of models to train
    models_to_train = [
        # ("VGG19", get_vgg19_model),
        # ("ViT", get_vit_model),
        ("AlexNet", get_alexnet_model),
        # ("ResNet50", get_resnet_model),
    ]
    
    for model_name, model_func in models_to_train:
        print(f"Training {model_name} model...")
        model = model_func()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_model(model, model_name, dataloader, criterion, optimizer)
    
    wandb.finish()

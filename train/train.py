import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from config import DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE
from data.dataset import ImageDataset
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the dataset
    dataset = ImageDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
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

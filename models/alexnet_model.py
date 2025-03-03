import torch.nn as nn
from torchvision import models
import wandb


def get_alexnet_model():
    model = models.alexnet(pretrained=True)
    in_features = model.classifier[6].in_features
    # Freeze all layers in the feature extractor
    if wandb.config.ALEX_FREEZE_FEATURES:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 4096),  # First FC layer
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4096, 512),  # Second FC layer
        nn.ReLU(),
        nn.Linear(
            512, len(wandb.config.NUM_CLASSES)
        ),  # Final output layer with correct num_labels
    )
    # model.classifier[6] = nn.Linear(in_features, wandb.config.NUM_CLASSES)
    return model

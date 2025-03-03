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
    model.classifier[6] = nn.Linear(in_features, wandb.config.NUM_CLASSES)
    return model

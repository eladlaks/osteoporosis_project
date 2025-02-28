import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES,ALEX_FREEZE_FEATURES

def get_alexnet_model():
    model = models.alexnet(pretrained=True)
    in_features = model.classifier[6].in_features
    # Freeze all layers in the feature extractor
    if ALEX_FREEZE_FEATURES:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
    return model

import torch.nn as nn
from torchvision import models
import wandb


def get_resnet_model():
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, wandb.config.NUM_CLASSES)
    return model

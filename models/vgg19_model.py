import torch.nn as nn
from torchvision import models
import wandb


def get_vgg19_model():
    model = models.vgg19(pretrained=True)
    # Replace the last layer to match the number of classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, wandb.config.NUM_CLASSES)
    return model

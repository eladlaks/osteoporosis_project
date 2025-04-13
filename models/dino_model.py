import torch.nn as nn
from torchvision import models
import wandb
from collections import OrderedDict
from timm import create_model  # Assuming timm library is used for DINOv2


def get_dinov2_model():
    # Load the DINOv2 model
    model = create_model('dinov2_base', pretrained=True)

    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Replace the classifier head
    num_ftrs = model.head.in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_ftrs, 512)),
                ("relu1", nn.ReLU()),
                ("dropout", nn.Dropout(0.5)),
                ("fc2", nn.Linear(512, wandb.config.NUM_CLASSES)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    model.head = classifier

    # Unfreeze the classifier layers
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True

    return model
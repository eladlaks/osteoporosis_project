import torch
import torch.nn as nn
from torchvision import models
import wandb
from collections import OrderedDict


def get_resnet_model(weights_path=None):
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")

    # Freeze all parameters initially
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_ftrs, 512)),
                ("relu1", nn.ReLU()),
                ("dropout", nn.Dropout(0.5)),
                ("fc2", nn.Linear(512, wandb.config.NUM_CLASSES)),
            ]
        )
    )
    model.fc = classifier
    
    # Unfreeze layer4 and the classifier for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    if weights_path:
        state_dict = torch.load(
            weights_path, map_location=wandb.config.DEVICE
        )  # or "cuda" if using GPU
        model.load_state_dict(state_dict)

    return model

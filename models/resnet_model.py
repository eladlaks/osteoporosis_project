import torch.nn as nn
from torchvision import models
import wandb
from collections import OrderedDict


def get_resnet_model():
    model = models.resnet50(pretrained=True)

    for parameter in model.parameters():
        parameter.requires_grad = False

    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc", nn.Linear(num_ftrs, wandb.config.NUM_CLASSES)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    model.fc = classifier
    return model

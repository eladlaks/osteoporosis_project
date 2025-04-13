from collections import OrderedDict
import torch
import torch.nn as nn
import timm
import wandb

def get_dinov2_model(num_classes=1000):
    # Load the DINOv2 meta model using timm
    # 'dinov2_base' is used as an example; adjust the model variant if necessary.
    backbone = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True)
    # Assume the model has a classifier head attribute named 'head'
    in_features = backbone.num_features
    # Remove the original classification head
    backbone.reset_classifier(0)
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(in_features, 512)),
                ("relu1", nn.ReLU()),
                ("dropout", nn.Dropout(0.5)),
                ("fc2", nn.Linear(512, wandb.config.NUM_CLASSES)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    # Create a new classifier for the target number of classes
    model = nn.Sequential(
        backbone,
        classifier
        )
    return model

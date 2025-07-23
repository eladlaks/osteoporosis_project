import torch
import torch.nn as nn
from torchvision import models
import wandb
from collections import OrderedDict
import timm                     # pip install timm


def get_timm_model(weights_path=None):
    """
    Returns model, feature_dim.
    name ∈ {"resnet34","resnet50","densenet121","efficientnet_b0"}
    """
    name = wandb.config.BACKBONE_NAME
    if name == "resnet34":
        model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, wandb.config.NUM_CLASSES)
    elif name == "resnet50":
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, wandb.config.NUM_CLASSES)
    elif name == "densenet121":
        model = models.densenet121(weights="DenseNet121_Weights.DEFAULT")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, wandb.config.NUM_CLASSES)
    elif name == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=wandb.config.NUM_CLASSES)
    else:
        raise ValueError("Unknown backbone")
    return model

def get_resnet_model(weights_path=None):

    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")

    # for parameter in model.parameters():
    #     parameter.requires_grad = False

    num_ftrs = model.fc.in_features
    # classifier = nn.Sequential(
    #     OrderedDict(
    #         [
    #             ("fc", nn.Linear(num_ftrs, wandb.config.NUM_CLASSES)),
    #             ("output", nn.LogSoftmax(dim=1)),
    #         ]
    #     )
    # )
    # model.fc = classifier

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_ftrs, 512)),
                ("relu1", nn.ReLU()),
                ("dropout", nn.Dropout(wandb.config.DROPOUT)),
                ("fc2", nn.Linear(512, wandb.config.NUM_CLASSES)),
            ]
        )
    )
    model.fc = classifier
    for name, param in model.named_parameters():
        # Unfreeze layer4 and the classifier
        param.requires_grad = False
        for layer in wandb.config.RESNET_LAYERS_TO_TRAIN:
            if layer in name:
                param.requires_grad = True
    if weights_path:
        state_dict = torch.load(
            weights_path, map_location=wandb.config.DEVICE
        )  # or "cuda" if using GPU
        model.load_state_dict(state_dict)

    return model

import torch
import torch.nn as nn
from torchvision import models
import wandb
from collections import OrderedDict
import timm  # pip install timm


def get_timm_model(weights_path=None, name=None):
    """
    Returns model, feature_dim.
    name ∈ {"resnet34","resnet50","densenet121","efficientnet_b0"}
    """

    num_classes = wandb.config.NUM_CLASSES
    dropout = wandb.config.DROPOUT
    if name == None:
        name = wandb.config.MODEL_NAME
    if name == "resnet34":
        # Freeze all parameters initially
        model = models.resnet34(weights="ResNet34_Weights.DEFAULT")
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layers in the last ResNet block (layer4) and the fully connected layers
        for param in model.layer4.parameters():
            param.requires_grad = True  
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(num_ftrs, 512)),
                    ("bn1", nn.BatchNorm1d(512)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(512, num_classes)),
                ]
            )
        )
        
    elif name == "resnet50":
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        num_ftrs = model.fc.in_features
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layers in the last ResNet block (layer4) and the fully connected layers
        for param in model.layer4.parameters():
            param.requires_grad = True 
        model.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(num_ftrs, 1024)),
                    ("bn1", nn.BatchNorm1d(1024)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(1024, 512)),
                    ("bn2", nn.BatchNorm1d(512)),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(dropout)),
                    ("fc3", nn.Linear(512, num_classes)),
                ]
            )
        )

    elif name == "densenet121":
        model = models.densenet121(weights="DenseNet121_Weights.DEFAULT")

        for name, param in model.features.named_parameters():
            if not name.startswith("denseblock4") and not name.startswith("denseblock3"):  # keep last dense block trainable
                param.requires_grad = False

        # Replace the classifier
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(num_ftrs, 512)),
                    ("bn1", nn.BatchNorm1d(512)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(512, num_classes)),
                ]
            )
        )


    elif name == "efficientnet_b0" or name == "efficientnet":
        model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

    else:
        raise ValueError("Unknown backbone")

    return model


def get_resnet_model(weights_path=None):

    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")

    # Freeze all parameters initially
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_ftrs, 1024)),
                ("bn1", nn.BatchNorm1d(1024)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(wandb.config.DROPOUT)),
                ("fc2", nn.Linear(1024, 512)),
                ("bn2", nn.BatchNorm1d(512)),
                ("relu2", nn.ReLU()),
                ("dropout2", nn.Dropout(wandb.config.DROPOUT)),
                ("fc3", nn.Linear(512, wandb.config.NUM_CLASSES)),
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

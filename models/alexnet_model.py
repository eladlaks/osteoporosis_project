import torch.nn as nn
from torchvision import models
import wandb


def get_alexnet_model():
    # model = models.alexnet(pretrained=True)
    # in_features = model.classifier[6].in_features
    # # Freeze all layers in the feature extractor
    # if wandb.config.ALEX_FREEZE_FEATURES:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    # model.classifier[6] = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(in_features, 4096),  # First FC layer
    #     nn.ReLU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(4096, 512),  # Second FC layer
    #     nn.ReLU(),
    #     nn.Linear(
    #         512, wandb.config.NUM_CLASSES
    #     ),  # Final output layer with correct num_labels
    # )
    ## model.classifier[6] = nn.Linear(in_features, wandb.config.NUM_CLASSES)

    # Load the pretrained AlexNet model
    model = models.alexnet(pretrained=True)

    # Freeze the first few layers in the feature extractor
    # For example, freeze the first 5 layers (adjust the index as needed)
    for idx, child in enumerate(model.features.children()):
        if idx < 5:
            for param in child.parameters():
                param.requires_grad = False

    # Now modify the classifier for your specific task
    in_features = model.classifier[6].in_features

    # Replace the final classifier layers with your custom classifier
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 4096),  # First FC layer
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(4096, 512),  # Second FC layer
        nn.ReLU(inplace=True),
        nn.Linear(
            512, wandb.config.NUM_CLASSES
        ),  # Final output layer with your number of classes
    )

    return model

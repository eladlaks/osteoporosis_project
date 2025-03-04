import torch.nn as nn
from torchvision import models
import wandb


def get_gideon_alexnet_model():
    model = models.alexnet(pretrained=True)

    # Ensure we get the correct input size for the first fully connected (fc) layer
    num_ftrs = model.classifier[1].in_features  # This should be 9216

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 4096),  # First FC layer
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4096, 512),  # Second FC layer
        nn.ReLU(),
        nn.Linear(
            512, wandb.config.NUM_CLASSES
        ),  # Final output layer with correct num_labels
    )

    return model

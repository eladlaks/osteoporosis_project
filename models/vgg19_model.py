import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_vgg19_model():
    model = models.vgg19(pretrained=True)
    # Replace the last layer to match the number of classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
    return model

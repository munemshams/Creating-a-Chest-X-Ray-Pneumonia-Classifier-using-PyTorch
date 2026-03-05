import torch.nn as nn
from torchvision import models

def build_model():

    model = models.resnet18(pretrained=True)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features

    # Replace final layer for binary classification
    model.fc = nn.Linear(num_features, 2)

    return model

"""The networks of the project, built from PyTorch layers."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MLP(nn.Module):
    """Feed-forward network: flatten the image, `depth` hidden layers of `width` units, one score per class."""

    def __init__(self, n_in, n_classes=5, width=512, depth=2, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        layers, n = [nn.Flatten()], n_in
        for _ in range(depth):
            layers += [nn.Linear(n, width), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n = width
        layers.append(nn.Linear(n, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)              # raw logits: nn.CrossEntropyLoss applies the softmax


class ResNetClassifier(nn.Module):
    """ResNet-18 pretrained on ImageNet, with a new head (random weights) of n_classes outputs."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def train(self, mode=True):
        # fit() calls model.train(): we stay in eval mode so that batch norm keeps its ImageNet
        # statistics, as in TP07 (ResNet-18 has no dropout, so nothing else changes)
        return super().train(False)

    def forward(self, x):
        return self.resnet(x)

    def features(self, x):
        """The 512 features of the last average pooling, just before the head."""
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x = r.layer4(r.layer3(r.layer2(r.layer1(x))))
        return torch.flatten(r.avgpool(x), 1)


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

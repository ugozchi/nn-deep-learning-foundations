"""The networks of the project, built from PyTorch layers."""
import torch.nn as nn


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


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

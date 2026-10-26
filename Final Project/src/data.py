"""Data shared by every model: the WikiArt subset, one stratified split and the DataLoaders.

The baseline, the MLP and the ResNet all use the same seed, so they are compared on the same
train / val / test images. From a notebook in `notebooks/`:

    import sys; sys.path.append("..")
    from src.data import make_loaders
    train_loader, val_loader, test_loader, classes = make_loaders("train", size=64)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms.functional import pil_to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "wikiart_style5"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_images(data_dir=DATA_DIR, size=128):
    """Every image as a (N, 3, size, size) tensor in [0, 1], the labels (N,) and the class names."""
    folder = datasets.ImageFolder(data_dir)            # one sub-folder per style, classes sorted by name
    X = torch.stack([pil_to_tensor(img) for img, _ in folder]).float() / 255
    if size != X.shape[-1]:
        X = F.interpolate(X, size=size, mode="bilinear", antialias=True, align_corners=False)
    y = torch.tensor(folder.targets)
    return X, y, folder.classes


def load_splits(normalisation="train", size=128, seed=0, data_dir=DATA_DIR):
    """Stratified 70 / 15 / 15 split, normalised per colour channel.

    normalisation="train": mean and std of the training images only (MLP, baseline).
    normalisation="imagenet": the statistics the pretrained ResNet was trained with.
    Returns {"train": (X, y), "val": (X, y), "test": (X, y)} and the class names.
    """
    X, y, classes = load_images(data_dir, size)
    train_idx, rest_idx = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=seed)

    if normalisation == "train":
        mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)    # statistics of the TRAINING images only
        std = X[train_idx].std(dim=(0, 2, 3), keepdim=True)
    elif normalisation == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise ValueError(f"unknown normalisation: {normalisation}")
    X = (X - mean) / std

    splits = {"train": (X[train_idx], y[train_idx]),
              "val": (X[val_idx], y[val_idx]),
              "test": (X[test_idx], y[test_idx])}
    for name, (_, y_split) in splits.items():
        counts = torch.bincount(y_split, minlength=len(classes)).tolist()
        print(f"{name:5s} {len(y_split):4d} images | per class {dict(zip(classes, counts))}")
    return splits, classes


def make_loaders(normalisation="train", size=128, batch_size=64, seed=0, data_dir=DATA_DIR):
    """The train / val / test DataLoaders (only the training one is shuffled) and the class names."""
    splits, classes = load_splits(normalisation, size, seed, data_dir)
    train_loader = DataLoader(TensorDataset(*splits["train"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*splits["val"]), batch_size=256)
    test_loader = DataLoader(TensorDataset(*splits["test"]), batch_size=256)
    return train_loader, val_loader, test_loader, classes

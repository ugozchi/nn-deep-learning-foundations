"""Training loop and metrics shared by every network, written from PyTorch primitives (TP04, TP06).

    torch.manual_seed(0)
    model = ...
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, n_epochs=30, patience=5)
    save_json(history, RESULTS_DIR / "mlp_history.json")
    metrics = classification_metrics(*predict(model, test_loader))
"""
import copy
import json
import time
import warnings
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def train_one_epoch(model, loader, loss_fn, optimizer, device="cpu"):
    """One pass over the training set; returns the mean loss and the accuracy measured on the fly."""
    model.train()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                 # 1. forward
        loss = loss_fn(logits, yb)         # 2. loss
        optimizer.zero_grad()              # 3. reset the gradients
        loss.backward()                    # 4. backward
        optimizer.step()                   # 5. update
        total += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return total / n, correct / n


def evaluate(model, loader, loss_fn, device="cpu"):
    """Mean loss and accuracy on a loader, in eval mode and without gradients."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return total / n, correct / n


def fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, patience=None,
        scheduler=None, device="cpu", seed=0, verbose=True):
    """Train for n_epochs, then reload the weights of the epoch with the lowest validation loss.

    patience: stop when the validation loss has not improved for `patience` epochs (None: never stop early).
    scheduler: stepped once per epoch, with the validation loss if it is a ReduceLROnPlateau.
    Returns the history: one value per epoch, plus the best epoch and the training time in seconds.
    best_epoch is 0 if the validation loss was never finite (e.g. NaN): the last weights are kept.
    """
    torch.manual_seed(seed)                # same order of the mini-batches for every run
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)       # this one needs the metric it watches
        elif scheduler is not None:
            scheduler.step()
        if verbose:
            print(f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f}")

        if val_loss < best_loss:           # early stopping (TP06): remember the best weights
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if patience is not None and wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:                                  # NaN from the first epoch (learning rate too high?)
        warnings.warn("the validation loss was never finite: keeping the last weights")
    history["best_epoch"] = best_epoch
    history["train_time_s"] = time.perf_counter() - start
    return history


def predict(model, loader, device="cpu"):
    """True and predicted labels over a whole loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            y_pred.append(model(xb.to(device)).argmax(dim=1).cpu())
            y_true.append(yb)
    return torch.cat(y_true), torch.cat(y_pred)


def classification_metrics(y_true, y_pred):
    """Accuracy, macro-F1 and confusion matrix as plain numbers (works for sklearn predictions too)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),   # rows: true class, columns: predicted
    }


def save_json(obj, path):
    """Save a history or a metrics dict, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

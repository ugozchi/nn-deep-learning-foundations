"""Figures shared by the notebooks."""
import matplotlib.pyplot as plt


def plot_history(history, title="", path=None):
    """Loss and accuracy per epoch, train vs validation; the dotted line marks the best epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, key, label in [(axes[0], "loss", "loss"), (axes[1], "acc", "accuracy")]:
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        if "best_epoch" in history:
            ax.axvline(history["best_epoch"], color="grey", linestyle=":", label="best epoch")
        ax.set_xlabel("epoch"); ax.set_ylabel(label); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    plt.show()


def plot_compare(histories, keys=("train_loss", "val_loss"), title="", path=None, log=False):
    """One panel per quantity, one curve per run (e.g. one run per learning rate)."""
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for name, history in histories.items():
            ax.plot(range(1, len(history[key]) + 1), history[key], label=name)
        ax.set_xlabel("epoch"); ax.set_ylabel(key.replace("_", " ")); ax.legend()
        if log:
            ax.set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=120)
    plt.show()

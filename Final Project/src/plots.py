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

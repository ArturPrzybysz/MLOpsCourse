from typing import List

import matplotlib.pyplot as plt

from src.paths import FIGURES_PATH


def plot_losses(train_loss: List[float],
                test_loss: List[float],
                title: str,
                file_name: str):
    epochs = list(range(len(train_loss)))

    plt.plot(epochs, train_loss, "g")
    plt.plot(epochs, test_loss, "r")
    plt.title(title)
    plt.legend(["train losses", "test losses"])
    plt.savefig(FIGURES_PATH / file_name)

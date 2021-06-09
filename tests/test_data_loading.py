import numpy as np

from src.data.dataset_loaders import prepare_MNIST_loaders
from src.paths import DATA_PATH


def test_MNIST_loading():
    train_loader, test_loader = prepare_MNIST_loaders(DATA_PATH, batch_size=1)

    train_len = np.sum([1 for _ in train_loader])
    test_len = np.sum([1 for _ in test_loader])

    assert train_len == 60000 and 60000 == len(train_loader)
    assert test_len == 10000 and 10000 == len(test_loader)

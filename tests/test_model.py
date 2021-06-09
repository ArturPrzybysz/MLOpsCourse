import torch
from src.data.dataset_loaders import prepare_MNIST_loaders
from src.models.train_model import Classifier
from src.paths import MODELS_PATH, DATA_PATH


def test_forward_pass():
    train_loader, test_loader = prepare_MNIST_loaders(DATA_PATH, batch_size=1)

    a = iter(test_loader).next()
    model = Classifier()
    state_dict = torch.load(MODELS_PATH / 'checkpoint.pth')
    model.load_state_dict(state_dict)
    pred = model(a[0])

    assert True

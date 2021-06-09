import torch

from src.models.train_model import Classifier
from src.paths import MODELS_PATH


def load_model() -> Classifier:
    model: Classifier = torch.load(MODELS_PATH / "checkpoint.pth")
    return model

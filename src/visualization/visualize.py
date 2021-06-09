import torch

from sklearn.manifold import TSNE

from src.models.predict_model import file_to_tensor
from src.models.train_model import Classifier
from src.paths import MODELS_PATH, DATA_PATH


def get_activation(param):
    pass


def visualise_hidden_activations():
    model: Classifier = torch.load(MODELS_PATH / "checkpoint.pth")
    model.register_forward_hook(get_activation('fc3'))
    img = file_to_tensor(DATA_PATH / "external" / "1.jpeg")
    out = model(img)
    out.activations["fc3"]

if __name__ == '__main__':
    pass

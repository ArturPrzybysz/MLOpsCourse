from pathlib import Path

from src.paths import MODELS_PATH, DATA_PATH
from train_model import Classifier  # required to torch.load
import torch
from PIL import Image
import numpy as np



def file_to_tensor(path, size=(28, 28)) -> torch.Tensor:
    image = Image.open(str(path))
    image.thumbnail(size, Image.ANTIALIAS)  # inplace
    img_np = np.array(image)
    img_2d = img_np.mean(axis=2)
    # plt.imshow(img_2d)
    # plt.show()
    img_torch = torch.tensor(img_2d)
    img_torch = torch.reshape(img_torch, (1, size[0], size[1]))
    return img_torch.float()


def predict(data_path: Path):
    model: Classifier = torch.load(MODELS_PATH / "checkpoint.pth")

    for img_path in data_path.glob("*g"):
        img = file_to_tensor(img_path)
        x = model(img)
        predictions = torch.exp(x)
        value, cls = predictions.topk(1)
        print(cls)


if __name__ == '__main__':
    predict(DATA_PATH / "external")

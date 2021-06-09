import torch
from torch import nn, optim
import torch.nn.functional as F

from reports.plots import plot_losses
from src.data.dataset_loaders import prepare_MNIST_loaders
from src.paths import MODELS_PATH, DATA_PATH


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


def train(testloader, trainloader, epochs=5):
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        N = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            log_ps = model(images)
            train_loss = criterion(log_ps, labels)
            train_loss.backward()
            optimizer.step()

            ps = torch.exp(model(images))

            top_p, top_class = ps.topk(k=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_loss += train_loss.item()
            N += len(labels)

        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(k=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy = torch.mean(equals.type(torch.FloatTensor))
                test_loss = criterion(log_ps, labels)
                test_losses.append(test_loss.item() / len(labels))
                train_losses.append(running_loss / N)

        print(f'Epoch {e}')
        print(f'Train accuracy: {accuracy.item() * 100}%')
        print(f'Test accuracy: {test_accuracy.item() * 100}%')
    return model, (train_losses, test_losses)


if __name__ == '__main__':
    train_loader, test_loader = prepare_MNIST_loaders(DATA_PATH)

    model, losses = train(test_loader, train_loader, epochs=1)
    plot_losses(losses[0], losses[1], "Train vs test loss MNIST", "mnist_losses.jpg")

    torch.save(model.state_dict(), MODELS_PATH / 'checkpoint.pth')

    model2 = Classifier()
    state_dict = torch.load(MODELS_PATH / 'checkpoint.pth')
    model2.load_state_dict(state_dict)

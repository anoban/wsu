# a stripped down MNIST classifier example from https://github.com/pytorch/examples/blob/main/mnist/main.py

from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, max_pool2d, nll_loss, relu
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lib.idx import IdxDataset


class Net(nn.Module):
    """ """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = log_softmax(x, dim=1)
        return output

    @override
    def train(self, train_loader: DataLoader[torch.Tensor], optimizer: Optimizer) -> None:
        for data, label in train_loader:
            optimizer.zero_grad()
            output = self.forward(data)
            loss = nll_loss(output, label)
            loss.backward()
            optimizer.step()

    def test(self, test_loader: DataLoader[torch.Tensor]) -> torch.Tensor:
        """ """

        self.eval()

        test_loss: float = 0.000
        correct: float = 0.000

        with torch.no_grad():
            for data, label in test_loader:
                data, label = data, label
                output = model(data)
                test_loss += nll_loss(output, label, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
            )
        )


def main() -> None:
    trainloader = DataLoader(
        dataset=IdxDataset(idx3_filepath=r"./MNIST/train-images-idx3-ubyte", idx1_filepath=r"./MNIST/train-labels-idx1-ubyte"),
        batch_size=1,
        num_workers=8,
    )
    testloader = DataLoader(
        dataset=IdxDataset(idx3_filepath=r"./MNIST/t10k-images-idx3-ubyte", idx1_filepath=r"./MNIST/t10k-labels-idx1-ubyte"),
        batch_size=1,
        num_workers=8,
    )

    model = Net()
    train(model, trainloader)


if __name__ == r"__main__":
    main()

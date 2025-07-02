# a stripped down MNIST classifier example from https://github.com/pytorch/examples/blob/main/mnist/main.py

from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, max_pool2d, nll_loss, relu
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from lib import IdxDataset


class Net(nn.Module):
    """ """

    def __init__(self):
        """ """

        super(Net, self).__init__()  # type: ignore - pyright keeps bitching

        self.conv_01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1)
        self.conv_02 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self.dropout_01 = nn.Dropout(0.25)
        self.dropout_02 = nn.Dropout(0.5)
        self.fucon_01 = nn.Linear(in_features=9216, out_features=128)
        self.fucon_02 = nn.Linear(in_features=128, out_features=10)

    @override
    def forward(self, _image: torch.Tensor) -> torch.Tensor:
        """
        carry out the forward pass of the data through the layers of the CNN, applying activation functions where appropriate.
        """

        _image = self.conv_01(_image)
        _image = relu(_image)
        _image = self.conv_02(_image)
        _image = relu(_image)
        _image = max_pool2d(_image, 2)
        _image = self.dropout_01(_image)
        _image = torch.flatten(_image, 1)
        _image = self.fucon_01(_image)
        _image = relu(_image)
        _image = self.dropout_02(_image)
        _image = self.fucon_02(_image)
        output = log_softmax(_image, dim=1)
        return output

    def fit(self, train_loader: DataLoader[torch.Tensor], optimizer: Optimizer) -> None:
        """
        train the convolutional neural network with the provided dataset and optimizer.
        """

        super().train(mode=True)  # set the module on training mode

        for data, label in train_loader:
            optimizer.zero_grad()
            output = self.forward(data)
            loss = nll_loss(output, label)
            loss.backward()
            optimizer.step()

    @torch.no_grad  # type: ignore
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """ """

        return self.forward(x).argmax()

    @torch.no_grad  # type: ignore
    def evaluate(self, test_loader: DataLoader[torch.Tensor]) -> tuple[float, float]:
        """
        retruns tuple[float, float] - (average loss, accuracy score)
        """

        super().eval()  # set the module on evaluation mode

        average_loss: float = 0.000
        accuracy_score: float = 0.000

        for data, label in test_loader:
            data, label = data, label
            output = self.forward(data)
            average_loss += nll_loss(output, label, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy_score += pred.eq(label.view_as(pred)).sum().item()

        # averaging
        average_loss /= len(test_loader.dataset)
        accuracy_score /= len(test_loader.dataset)

        return (average_loss, accuracy_score)


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
    model.fit(trainloader, SGD(params=model.parameters(), lr=0.001))
    model.evaluate(testloader)


if __name__ == r"__main__":
    main()

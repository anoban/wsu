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
        super(Net, self).__init__()  # type: ignore

        self.conv_01 = nn.Conv2d(1, 32, 3, 1)
        self.conv_02 = nn.Conv2d(32, 64, 3, 1)
        self.dropout_01 = nn.Dropout(0.25)
        self.dropout_02 = nn.Dropout(0.5)
        self.fcon_01 = nn.Linear(9216, 128)
        self.fcon_02 = nn.Linear(128, 10)

    @override
    def forward(self, _image: torch.Tensor) -> torch.Tensor:
        _image = self.conv_01(_image)
        _image = relu(_image)
        _image = self.conv_02(_image)
        _image = relu(_image)
        _image = max_pool2d(_image, 2)
        _image = self.dropout_01(_image)
        _image = torch.flatten(_image, 1)
        _image = self.fcon_01(_image)
        _image = relu(_image)
        _image = self.dropout_02(_image)
        _image = self.fcon_02(_image)
        output = log_softmax(_image, dim=1)
        return output

    def fit(self, train_loader: DataLoader[torch.Tensor], optimizer: Optimizer) -> None:
        """ """

        super().train(mode=True)  # set the module on training mode

        for data, label in train_loader:
            optimizer.zero_grad()
            output = self.forward(data)
            loss = nll_loss(output, label)
            loss.backward()
            optimizer.step()

    @torch.no_grad  # type: ignore
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax()

    @torch.no_grad  # type: ignore
    def evaluate(self, test_loader: DataLoader[torch.Tensor]) -> tuple[float, float]:
        """
        retruns tuple[float, float] - (average loss, accuracy score)
        """

        super().eval()  # set the module on evaluation mode

        test_loss: float = 0.000
        correct: float = 0.000

        for data, label in test_loader:
            data, label = data, label
            output = self.forward(data)
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
    model.fit(trainloader)
    model.evaluate(testloader)


if __name__ == r"__main__":
    main()

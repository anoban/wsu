from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, max_pool2d, nll_loss, relu
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from lib import IdxDataset

__doc__ = r"A collection of CNN classifiers with different architecures for MNIST style datasets"
__all__ = (r"CNN0",)


class CNN0(nn.Module):
    """"""

    def __init__(self) -> None:
        super(CNN0, self).__init__()


class CNN1(nn.Module):
    """
    Architecture:

    """

    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        """ """

        super(CNN1, self).__init__()  # type: ignore - pyright keeps bitching about __init__()'s type
        self.__nchannels = n_channels  # number of colour channels
        self.__nclasses = n_classes  # number of image classes

        # convolution layers
        self._conv_01 = nn.Conv2d(
            in_channels=self.__nchannels, out_channels=8, kernel_size=(4, 4), stride=1
        )  # using a 4 x 4 kernel since our images will be 28 x 28
        self._conv_02 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(4, 4), stride=1
        )  # in channels of the next layer should match the out channels of the previous layer

        # fully connected layers
        self._fcon_01 = nn.Linear(
            in_features=16
            * 7
            * 7,  # last output from the second convolutional layer will have 40 channels, after the two pooling transformations, we'll have 7 x 7 matrices for images
            out_features=64,
        )
        self._fcon_02 = nn.Linear(in_features=64, out_features=32)
        self._fcon_03 = nn.Linear(in_features=32, out_features=self.__nclasses)

    @override
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """ """

        super().train(mode=True)  # set the base class on training mode
        _input = self._conv_01(_input)  # apply the first convolution operation
        _input = relu(_input)  # activation
        _input = nn.MaxPool2d(kernel_size=(2, 2), stride=4)(_input)  # apply max pooling, 28 x 28 matrices will become 14 x 14 matrices

        _input = self._conv_02(_input)  # apply the second convolution
        _input = relu(_input)  # activation
        _input = nn.AvgPool2d(kernel_size=(2, 2), stride=4)(_input)  # apply average pooling, 14 x 14 matrices will become 7 x 7 matrices

        # pass the result through the fully connected layers
        _input = self._fcon_01(_input)
        _input = self._fcon_02(_input)
        _input = self._fcon_03(_input)

        return _input

    def fit(self) -> None:
        pass


class CNN2(nn.Module):
    """
    a stripped down MNIST classifier example from https://github.com/pytorch/examples/blob/main/mnist/main.py

    Architecture:

    """

    def __init__(self, n_channels: int = 1, n_classes: int = 10):
        """ """

        super(CNN2, self).__init__()  # type: ignore

        self._conv_01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1)
        self._conv_02 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1)
        self._fucon_01 = nn.Linear(in_features=9216, out_features=128)
        self._fucon_02 = nn.Linear(in_features=128, out_features=10)

    @override
    def forward(self, _image: torch.Tensor) -> torch.Tensor:
        """
        carry out the forward pass of the data through the layers of the CNN, applying activation functions where appropriate.
        """

        _image = self._conv_01(_image)
        _image = relu(_image)
        _image = self._conv_02(_image)
        _image = relu(_image)
        _image = max_pool2d(_image, 2)
        _image = nn.Dropout(0.25)(_image)
        _image = torch.flatten(_image, 1)
        _image = self._fucon_01(_image)
        _image = relu(_image)
        _image = nn.Dropout(0.5)(_image)
        _image = self._fucon_02(_image)
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

        super().eval()  # set the base class on evaluation mode, equivalent to super().train(mode=False)
        return self.forward(x).argmax()

    @torch.no_grad  # type: ignore
    def evaluate(self, test_loader: DataLoader[torch.Tensor]) -> tuple[float, float]:
        """
        retruns tuple[float, float] - (average loss, accuracy score)
        """

        super().eval()  # set base class module on evaluation mode, equivalent to super().train(mode=False)

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
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = CNN1(n_channels=1, n_classes=10)

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()

    for data, label in train_loader:
        print(data.shape)
        out = model(data)

        loss = criterion(out, label)
        loss.backward()


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

    model = CNN2()
    model.fit(trainloader, SGD(params=model.parameters(), lr=0.001))
    model.evaluate(testloader)


if __name__ == r"__main__":
    main()


if __name__ == r"__main__":
    main()

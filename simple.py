from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import SGD
from torch.utils.data import DataLoader

from idx import IdxDataset


class SimNet(nn.Module):
    """ """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        """ """

        super(SimNet, self).__init__()
        self.__nchannels = n_channels  # number of colour channels
        self.__nclasses = n_classes  # number of image classes

        # convolution layers
        self.__conv_01 = nn.Conv2d(
            in_channels=self.__nchannels, out_channels=8, kernel_size=(4, 4), stride=1
        )  # using a 4 x 4 kernel since our images are 28 x 28

        # pooling layers
        self.__maxpool = nn.MaxPool2d(kernel_size=(4, 4), stride=4)  # a 28 x 28 image will be transformed into a 7 x 7 image

        # fully connected layers
        self.__fcon_01 = nn.Linear(
            in_features=8
            * 7
            * 7,  # output from the convolutional layer will have 40 channels, after the two pooling transformations, we'll have 7 x 7 matrices for images
            out_features=16,
        )
        self.__fcon_02 = nn.Linear(in_features=16, out_features=self.__nclasses)

    @override
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """ """

        _input = self.__conv_01(_input)  # apply the convolution
        _input = relu(_input)  # activation
        _input = self.__maxpool(_input)  # apply max pooling, 28 x 28 matrices will become 14 x 14 matrices
        # pass the result through the fully connected layer
        _input = self.__fcon_01(_input)
        _input = self.__fcon_02(_input)

        return _input


def main() -> None:
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = SimNet(n_channels=1, n_classes=10)

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()

    for data, label in train_loader:
        out = model(data)
        loss = criterion(out, label)
        loss.backward()


if __name__ == r"__main__":
    main()

from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, relu
from torch.optim import SGD
from torch.utils.data import DataLoader

from lib.idx import IdxDataset


class SimpleNNet(nn.Module):
    """ """

    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        """
        n_channels: int - number of colour channels in the input images
        n_classes: int - number of image classes
        """

        super(SimpleNNet, self).__init__()
        self.__nchannels = n_channels
        self.__nclasses = n_classes

        # first convolution layer, a 28 x 28 image becomes a 26 x 26 image????
        self.__conv_01 = nn.Conv2d(in_channels=self.__nchannels, out_channels=24, kernel_size=(3, 3), stride=1)

        # second convolution layer, the 26 x 26 image becomes a 24 x 24 image???
        self.__conv_02 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), stride=1)

        # pooling layer
        self.__maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # a 24 x 24 image will be transformed into a 12 x 12 image

        # first fully connected layer, output from the convolutional layer will have 48 channels, after max pooling, we'll have 12 x 12 matrices for images
        self.__fcon_01 = nn.Linear(in_features=144, out_features=24)

        # second fully connected layer
        self.__fcon_02 = nn.Linear(in_features=24, out_features=self.__nclasses)

    @override
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """ """

        image = self.__conv_01(image)  # apply the first convolution
        # image becomes a 24 x 26 x 26 tensor
        image = relu(image)  # activation

        image = self.__conv_02(image)  # apply the second convolution
        # image becomes a 48 x 24 x 24 tensor
        image = relu(image)  # activation

        image = self.__maxpool(image)  # apply max pooling, 24 x 24 matrices will become 12 x 12 matrices
        # image becomes a 48 x 12 x 12 tensor
        print(image.shape)

        # flatten the tensor i.e the 48 x 12 x 12 tensor will become a 48 x 144 matrix
        image = torch.flatten(input=image, start_dim=1)
        print(image.shape)

        # pass the result through the fully connected layers
        image = self.__fcon_01(image)
        print(image.shape)
        image = relu(image)  # activation
        image = self.__fcon_02(image)

        # apply softmax
        image = log_softmax(image, dim=1)

        return image


def main() -> None:
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = SimpleNNet(n_channels=1, n_classes=10)

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()

    for image, label in train_loader:
        out = model(image)
        loss = criterion(out, label)
        loss.backward()


if __name__ == r"__main__":
    main()

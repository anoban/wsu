from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, relu
from torch.optim import SGD
from torch.utils.data import DataLoader


class LiNN(nn.Module):
    """
    A fully connected linear neural network without any convolutions
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        """
        n_channels: int - number of colour channels in the input images
        n_classes: int - number of image classes
        """

        super(LiNN, self).__init__()  # type: ignore
        self.__nchannels = n_channels
        self.__nclasses = n_classes

        # first convolution layer, a 28 x 28 image becomes a 14 x 14 image
        self.__conv_01 = nn.Conv2d(in_channels=self.__nchannels, out_channels=4, kernel_size=(2, 2), stride=2)

        # second convolution layer, the 14 x 14 image becomes a 7 x 7 image
        self.__conv_02 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=(2, 2), stride=2)

        # pooling layer
        self.__maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # a 24 x 24 image will be transformed into a 12 x 12 image

        # first fully connected layer, output from the convolutional layer will have 48 channels, after max pooling, we'll have 12 x 12 matrices for images
        self.__fcon_01 = nn.Linear(in_features=144, out_features=24)

        # second fully connected layer
        self.__fcon_02 = nn.Linear(in_features=24, out_features=self.__nclasses)

    @override
    def forward(self, _image: torch.Tensor) -> torch.Tensor:
        """ """

        super().train(mode=True)
        _image = self.__conv_01(_image)  # apply the first convolution
        # image becomes a 24 x 26 x 26 tensor
        _image = relu(_image)  # activation

        _image = self.__conv_02(_image)  # apply the second convolution
        # image becomes a 48 x 24 x 24 tensor
        _image = relu(_image)  # activation

        _image = self.__maxpool(_image)  # apply max pooling, 24 x 24 matrices will become 12 x 12 matrices
        # image becomes a 48 x 12 x 12 tensor
        print(_image.shape)

        # flatten the tensor i.e the 48 x 12 x 12 tensor will become a 48 x 144 matrix
        _image = torch.flatten(input=_image, start_dim=1)
        print(_image.shape)

        # pass the result through the fully connected layers
        _image = self.__fcon_01(_image)
        print(_image.shape)
        _image = relu(_image)  # activation
        _image = self.__fcon_02(_image)

        # apply softmax
        _image = log_softmax(_image, dim=1)

        return _image

    def fit(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass


def main() -> None:
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = LiNN(n_channels=1, n_classes=10)

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()

    for image, label in train_loader:
        out = model(image)
        loss = criterion(out, label)
        loss.backward()


if __name__ == r"__main__":
    main()

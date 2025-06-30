from typing import override

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn.functional import relu
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset


class IdxDataset(Dataset[torch.uint8]):
    """
    A PyTorch compatible class to handle IO and iteration operations with idx1 (labels) and Idx3 (images) files

    IDX1 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

    IDX3 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    """

    def __init__(self, idx1_filepath: str, idx3_filepath: str) -> None:
        """
        `Parameters`:
        idx1_filepath: str - path to unzipped idx1 resource (labels)
        idx3_filepath: str - path to unzipped idx3 resource (images)

        `Returns`:
        None

        `Notes`:
        Depends on NumPy and PyTorch
        """

        super(IdxDataset, self).__init__()

        try:  # OPEN THE IDX1 FILE
            with open(file=idx1_filepath, mode="rb") as fp:
                idx1_ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{idx1_filepath} is not found on this computer!") from fnf_error

        try:  # OPEN THE IDX3 FILE
            with open(file=idx3_filepath, mode="rb") as fp:
                idx3_ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{idx3_filepath} is not found on this computer!") from fnf_error

        self.__idx1_magic: int = int.from_bytes(idx1_ubytes[:4], byteorder="big")  # idx1 magic number
        self.__idx1_count: int = int.from_bytes(idx1_ubytes[4:8], byteorder="big")  # count of the data elements (labels)

        assert (self.__idx1_count == idx1_ubytes.size - 8) and (self.__idx1_magic == 2049), (
            f"There seems to be a parsing error or the binary file {idx1_filepath} is corrupted!"
        )

        self.__labels: torch.FloatTensor = torch.FloatTensor(
            idx1_ubytes[8:].astype(np.float32)
        )  # type casting the data from np.uint8 to np.float32 since np.exp() raises FloatingPointError with np.uint8 arrays

        self.__idx3_magic: int = int.from_bytes(idx3_ubytes[:4], byteorder="big")  # idx3 magic number
        self.__idx3_count: int = int.from_bytes(idx3_ubytes[4:8], byteorder="big")  # count of the data elements (images)
        self.__image_res: tuple[int, int] = (
            int.from_bytes(idx3_ubytes[8:12], byteorder="big"),
            int.from_bytes(idx3_ubytes[12:16], byteorder="big"),
        )  # shape of each element

        assert ((self.__idx3_count * self.__image_res[0] * self.__image_res[1]) == (idx3_ubytes.size - 16)) and (
            self.__idx3_magic == 2051
        ), f"There seems to be a parsing error or the binary file {idx3_filepath} is corrupted!"

        # idx3 file stores data as bytes but we'll load in each byte as a 32 bit floats because np.exp() raises a FloatingPointError with np.uint8 type arrays
        self.__data: torch.FloatTensor = torch.FloatTensor(
            idx3_ubytes[16:].reshape(self.__idx3_count, self.__image_res[0], self.__image_res[1]).astype(np.float32)
        )

        assert self.__idx1_count == self.__idx3_count, (
            f"The pair of Idx1 and Idx3 files passed seem incompatible!, {idx3_filepath} has {self.__idx3_count} images while {idx1_filepath} has {self.__idx1_count} labels!"
        )

    @override
    def __repr__(self) -> str:
        return f"Idx3 object (magic: {self.__magic}, shape: {self.__image_res}, count: {self.__count:,})"

    def __len__(self) -> int:
        return self.__idx3_count  # __idx3_count == __idx1_count, so ...

    @override
    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, torch.FloatType]:
        """
        `Parameters`:
        index: int - the column to return from the matrix.

        `Returns`:
        (torch.FloatTensor, torch.float64) - all pixels of index th image and it's corresponding label

        `Notes`:
        IndexErrors are left to PyTorch to handle.
        """

        return self.__data[index], self.__labels[index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("IdIdxDataset objects are immutable!")


class ConvNNet(nn.Module):
    """ """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        """ """

        super(ConvNNet, self).__init__()
        self.__nchannels = n_channels  # number of colour channels
        self.__nclasses = n_classes  # number of image classes

        # convolution layers
        self.__conv_01 = nn.Conv2d(
            in_channels=self.__nchannels, out_channels=8, kernel_size=(4, 4), stride=1
        )  # using a 4 x 4 kernel since our images will be 28 x 28
        self.__conv_02 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(4, 4), stride=1
        )  # in channels of the next layer should match the out channels of the previous layer

        # pooling layers
        self.__maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=4)  # a 28 x 28 image will be transformed into a 14 x 14 image
        self.__avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=4)  # a 14 x 14 image will be transformed into a 7 x 7 image

        # fully connected layers
        self.__fcon_01 = nn.Linear(
            in_features=16
            * 7
            * 7,  # last output from the second convolutional layer will have 40 channels, after the two pooling transformations, we'll have 7 x 7 matrices for images
            out_features=64,
        )
        self.__fcon_02 = nn.Linear(in_features=64, out_features=32)
        self.__fcon_03 = nn.Linear(in_features=32, out_features=self.__nclasses)

    @override
    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """ """

        _input = self.__conv_01(_input)  # apply the first convolution operation
        _input = relu(_input)  # activation
        _input = self.__maxpool(_input)  # apply max pooling, 28 x 28 matrices will become 14 x 14 matrices

        _input = self.__conv_02(_input)  # apply the second convolution
        _input = relu(_input)  # activation
        _input = self.__avgpool(_input)  # apply average pooling, 14 x 14 matrices will become 7 x 7 matrices

        # pass the result through the fully connected layers
        _input = self.__fcon_01(_input)
        _input = self.__fcon_02(_input)
        _input = self.__fcon_03(_input)

        return _input


def main() -> None:
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = ConvNNet(n_channels=1, n_classes=10)

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()

    for i, batch in enumerate(train_loader):
        data, labels = batch
        out = model(data)

        loss = criterion(out, labels)
        loss.backward()


if __name__ == r"__main__":
    main()

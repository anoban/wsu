from typing import override

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.nn.functional import relu
from torch.utils.data import Dataset


class Idx1(Dataset):
    """
    A minimal class to handle IO operations using idx1 files

    IDX1 file format:
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    """

    def __init__(self, filepath: str) -> None:
        """
        `Parameters`:
        filepath: str - path to unzipped idx1 resource

        `Returns`:
        None

        `Notes`:
        Depends on NumPy
        """
        super(Idx1, self).__init__()

        try:
            with open(file=filepath, mode="rb") as fp:
                ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)  # private
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{filepath} is not found on this computer!") from fnf_error

        self.__magic: int = int.from_bytes(ubytes[:4], byteorder="big")  # idx magic number
        self.__count: int = int.from_bytes(ubytes[4:8], byteorder="big")  # count of the data elements (labels)
        assert self.__count == ubytes.size - 8, "There seems to be a parsing error or the binary file is corrupted!"
        self.__data: torch.FloatTensor = torch.Tensor(
            ubytes[8:].astype(np.float32)
        )  # type casting the data from np.uint8 to np.float64 since np.exp() raises FloatingPointError with np.uint8 arrays

    @override
    def __repr__(self) -> str:
        return f"Idx1 object (magic: {self.__magic}, count: {self.__count:,})"

    @override
    def __len__(self) -> int:
        return self.__count

    @override
    def __getitem__(self, index: int) -> torch.float32:  #
        """
        `Parameters`:
        index: int - offset of the element to return from the labels array.

        `Returns`:
        np.float64 - the index th element in the labels array

        `Notes`:
        IndexErrors are left for NumPy to handle.
        """

        return self.__data[index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("Idx1 objects are immutable!")


class Idx3(Dataset):
    """
    A minimal class to handle IO operations using idx3 files

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

    def __init__(self, filepath: str) -> None:
        """
        `Parameters`:
        filepath: str - path to unzipped idx3 resource

        `Returns`:
        None

        `Notes`:
        Depends on NumPy
        """
        super(Idx3, self).__init__()

        try:
            with open(file=filepath, mode="rb") as fp:
                ubytes: NDArray[np.uint8] = np.fromfile(fp, dtype=np.uint8)  # private
        except FileNotFoundError as fnf_error:
            raise RuntimeError(f"{filepath} is not found on this computer!") from fnf_error

        self.__magic: int = int.from_bytes(ubytes[:4], byteorder="big")  # idx3 magic number
        self.__count: int = int.from_bytes(ubytes[4:8], byteorder="big")  # count of the data elements (images)
        self.__image_res: tuple[int, int] = (
            int.from_bytes(ubytes[8:12], byteorder="big"),
            int.from_bytes(ubytes[12:16], byteorder="big"),
        )  # shape of each element
        assert (self.__count * self.__image_res[0] * self.__image_res[1]) == (ubytes.size - 16), (
            "There seems to be a parsing error or the binary file is corrupted!"
        )

        # idx3 file stores data as bytes but we'll load in each byte as a 32 bit floats because np.exp() raises a FloatingPointError with np.uint8 type arrays
        self.__data: torch.FloatTensor = torch.Tensor(
            ubytes[16:].reshape(self.__count, self.__image_res[0], self.__image_res[1]).astype(np.float32)
        )

    @override
    def __repr__(self) -> str:
        return f"Idx3 object (magic: {self.__magic}, shape: {self.__image_res}, count: {self.__count:,})"

    @override
    def __len__(self) -> int:
        return self.__count

    @override
    def __getitem__(self, index: int) -> torch.FloatTensor:
        """
        `Parameters`:
        index: int - the column to return from the matrix.

        `Returns`:
        NDArray[np.float64] - all pixels of index th image, i.e returns the index th column of the transposed matrix

        `Notes`:
        IndexErrors are left for NumPy to handle.
        """

        return self.__data[index]

    def __setitem__(self, index: int) -> None:
        raise PermissionError("Idx3 objects are immutable!")


class ConvNNet(nn.Module):
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
    def forward(self, _input: torch.FloatTensor) -> torch.FloatTensor:
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
    train_x = Idx3(r"../FashionMNIST/train-images-idx3-ubyte")
    train_y = Idx1(r"../FashionMNIST/train-labels-idx1-ubyte")

    test_x = Idx3(r"../FashionMNIST/t10k-images-idx3-ubyte")
    test_y = Idx1(r"../FashionMNIST/t10k-labels-idx1-ubyte")

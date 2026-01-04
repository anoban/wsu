# helper classes and functions for finetuning SAM on our custom dataset

import os
from typing import Optional, override

import torch
import torchvision.transforms.v2 as transforms_v2  # type: ignore
from torch.utils.data import Dataset


class RootImagesDataset(Dataset):
    """
    Docstring for RootImagesDataset
    """

    @staticmethod
    def _apply_transforms(_transforms: transforms_v2.Compose, _obj: torch.Tensor) -> torch.Tensor:
        """
        Docstring for _apply_transforms

        :param _transforms: Description
        :type _transforms: transforms_v2.Compose
        :param _obj: Description
        :type _obj: torch.Tensor
        :return: Description
        :rtype: Tensor
        """
        pass

    def __init__(
        self, dir_images: str, dir_annotations: str, transformations: Optional[transforms_v2.Compose], pretransform: bool = True
    ) -> None:
        """
        Docstring for __init__

        :param self: Description
        :param dir_images: Description
        :type dir_images: str
        :param dir_annotations: Description
        :type dir_annotations: str
        :param transformations: Description
        :type transformations: Optional[transforms_v2.Compose]
        :param pretransform: Description
        :type pretransform: bool
        """

        super().__init__()
        # strip off the file extensions and save the base names
        images = set([im.replace(r".png", "") for im in os.listdir(dir_images)])
        annotations = set([an.replace(r".json", "") for an in os.listdir(dir_annotations)])

        # check whether all images have annotations - we expect the images to be PNG files (.png) and annotations to be JSON files (.json).
        # only pick the images that have annotations and leave the others
        self._items = list(images.intersection(annotations))
        if not len(self._items):  # if there's no matching images and annotations,
            raise RuntimeError(r"No matching image files and annotation files found in the provided directories!")

    def __len__(self) -> int:
        """
        Docstring for __len__

        :param self: Description
        :return: Description
        :rtype: int
        """

        return len(self._items)

    @override
    def __getitem__(self, index: int):
        pass

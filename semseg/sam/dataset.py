import os
from typing import Optional, override

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset


class RootImagesDataset(Dataset[torch.Tensor]):
    """ """

    @staticmethod
    def _apply_batch_transforms(_transforms: transforms_v2.Compose, _batch: torch.Tensor) -> torch.Tensor:
        """
        :param _transforms: transformations to be applied to the images or annotations
        :type _transforms: transforms_v2.Compose
        :param _batch: a batch of images or annotations
        :type _batch: torch.Tensor
        :return: transformed image or annotation batches
        :rtype: Tensor
        """
        return _transforms(_batch)  # feels like overcomplicating?????

    @staticmethod
    def _read_images_into_tensor(fnames: list[str]) -> torch.Tensor:
        """
        :param fnames: file names of the images
        :type fnames: list[str]
        :return: a 4D tensor of shape (n_imgs, width, height, n_clrchannels)
        :rtype: Tensor
        """

        imgs: list[NDArray[np.uint8]] = []
        for fname in fnames:
            try:
                with open(file=fname, mode="rb") as fp:
                    obj = Image.open(fp)  # opens in RGB colour chanel mode by default, unlike opencv, which is what we want
                    if obj.mode != "RGB":
                        obj = obj.convert(r"RGB")  # if the colour channel is not RGB, convert it to RGB
                    imgs.append(np.array(obj, dtype=np.uint8))
            except (PermissionError, FileNotFoundError) as excpt:
                raise RuntimeError(f"Filed to read file {fname}") from excpt
        return torch.Tensor(
            np.array([img for img in imgs])
        )  # PyTorch recommends converting the list of numpy arrays into an array of arrays before contructing a tensor for performance reasons

    @staticmethod
    def _read_annotations_into_tensor(fnames: list[str]) -> torch.Tensor:
        """
        :param fnames: Description
        :type fnames: list[str]
        :return: Description
        :rtype: Tensor
        """
        pass

    def __init__(
        self, dir_images: str, dir_annotations: str, transformations: Optional[transforms_v2.Compose] = None, pretransform_all: bool = True
    ) -> None:
        """
        :param dir_images: path to the directory that contains the PNG images
        :type dir_images: str
        :param dir_annotations: path to the directory that contains the annotation JSON files
        :type dir_annotations: str
        :param transformations: transformations to be applied to the images and annotations
        :type transformations: Optional[transforms_v2.Compose]
        :param pretransform_all: whether to apply the transformations to all the images and annotations during class instantiation instead of within each call to __getitem__
        :type pretransform_all: bool
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

        self._pretransform = pretransform_all
        self._transforms = transformations

    def __len__(self) -> int:
        """
        :return: length of the dataset (image and annotation pairs)
        :rtype: int
        """

        return len(self._items)

    @override
    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param self: Description
        :param _idx: Description
        :type _idx: int
        :return: Description
        :rtype: tuple[Tensor, Tensor]
        """
        pass

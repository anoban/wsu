# this file contains a subclass derived from torch.utils.data.Dataset and customized to be used with SAM, SAM 2.1 and MaskRCNN
# we can use this one  subclass for all the three models, should help streamline the finetuning

import json
import os
import warnings
from typing import Optional, override

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2  # pyright: ignore[reportMissingTypeStubs]
from PIL import Image
from skimage.draw import polygon2mask  # pyright: ignore[reportUnknownVariableType]
from torch.utils.data import Dataset


class RootImageDataset(Dataset[torch.Tensor]):
    """
    subclasses of Dataset don't need to handle batching and other stuff as these are usually handled by the DataLoader class
    reading in all the images and annotations and transforming them all at once, during class instance initialization could save us some time
    could collate all the images into one big tensor and all the annotations into another - will also make memry access more effcient
    but will definetely increase the memory use, opting to reading and transforming within __getitem__.

    references:
    SAM - https://www.labellerr.com/blog/fine-tune-sam-on-custom-dataset/
    SAM 2 - https://www.datacamp.com/tutorial/sam2-fine-tuning
    SAM 2 - https://learnopencv.com/finetuning-sam2/
    MaskRCNN - https://github.com/cylcharles/Pytorch_exercise/blob/master/Mask%20R-CNN%20finetuning_instance_segmentation.ipynb

    look into https://github.com/wkentaro/labelme/issues/777 for saving labelme annotations without embedding the binary image data
    in our photos of roots - segmentation classes will be roots, mycorrhizal hyphae and background (3 classes)
    """

    def __init__(self, dir_images: str, dir_annotations: str, transf: Optional[transforms_v2.Compose] = None) -> None:
        """
        https://docs.pytorch.org/vision/0.22/generated/torchvision.transforms.Compose.html

        :param dir_images: path to the directory that contains the PNG images
        :type dir_images: str
        :param dir_annotations: path to the directory that contains the annotation JSON files
        :type dir_annotations: str
        :param transformations: transformations to be applied to the images and annotations
        :type transformations: Optional[transforms_v2.Compose]
        """

        super().__init__()

        # strip off the file extensions and save the base names
        images = np.array(os.listdir(dir_images))  # photos from the WiFi microscope are .png s, not relaxing the extension restriction
        # assert all([img.endswith((".jpg", ".jpeg")) for img in images]), (
        #     "Contents of the image directory are expected to be in JFIF format!"
        # )
        self._img_extension: str = images[0].split(".")[1]  # capture the extensions used in the images

        annotations = np.array(os.listdir(dir_annotations))
        assert all([ann.endswith(".json") for ann in annotations]), "Contents of the annotation directory are expected to be JSON files!"

        # check whether all images have annotations - we expect the images to be PNG files (.png) and annotations to be JSON files (.json).
        # only pick the images that have annotations and leave the others
        self._matched_basenames = np.intersect1d([img.split(".")[0] for img in images], [ann.split(".")[0] for ann in annotations])

        if not self._matched_basenames.size:  # if there's no matching images and annotations,
            raise RuntimeError(r"No matching image files and annotation files found in the provided directories!")

        if len(self._matched_basenames) != len(annotations):  # if we don't have images for all the annotations
            warnings.warn(
                f"Mismatch in the contents of image ({dir_images}) and annotations ({dir_annotations}) directories has been detected!",
                category=RuntimeWarning,
            )

        self._transforms = transf
        self._image_dir = dir_images
        self._annot_dir = dir_annotations

    def __len__(self) -> int:
        """
        :return: length of the dataset (matched image and annotation pairs)
        :rtype: int
        """

        return self._matched_basenames.size

    @override
    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """ """

        path_img = os.path.join(self._image_dir, f"{self._matched_basenames[_idx]}.{self._img_extension}")
        path_ann = os.path.join(self._annot_dir, f"{self._matched_basenames[_idx]}.json")

        with open(file=path_img, mode="rb") as fp:  # let open() handle the errors
            # all the images will be from the camera of a Samsung A22 5G (JFIF images)
            img = Image.open(fp)  # opens in RGB colour chanel mode by default, unlike opencv, which is what we want
            if img.mode != "RGB":
                img = img.convert("RGB")  # if the colour channel is not RGB, convert it to RGB

        img = torch.tensor(np.array(img, dtype=np.float32))
        if self._transforms:
            img = self._transforms(img)  # if transforms have been specified, apply them

        # all the annotations will be from Labelme, which doesn't use any kind of compressions
        with open(file=path_ann, mode="rt") as fp:
            ann = json.load(fp=fp)

        bmasks = torch.tensor(
            np.array(  # and a single .json file will typically contain multiple segmentation masks
                [
                    polygon2mask(image_shape=(ann["imageWidth"], ann["imageHeight"]), polygon=polygon["points"])
                    .astype(np.uint8)
                    .T  # labelme annotations need to be transposed to match the raw images
                    for polygon in ann["shapes"]
                ]
            ),
            dtype=torch.float32,
        )

        return img, bmasks

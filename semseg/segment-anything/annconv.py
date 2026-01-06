# convert the raw JSON annotation files created by labelme to RLE compressed JSON files, which will be convenient when training on servers
# we do not want to upload huge JSON files, do we???
# and it has the whole image file embedded in it???

import json
from typing import Any

from pycocotools import mask as rlemask
from skimage.draw import polygon2mask  # pyright: ignore[reportUnknownVariableType]


class RLEAnnotation:
    """
    Docstring for RLEAnnotation
    """

    # LABELME_JSON_KEYS = ("version", "flags", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth")
    # we don't actually need a few keys: "version", "flags" & "imageData", for our use case
    LABELME_ESSENTIAL_JSON_KEYS = ("shapes", "imagePath", "imageHeight", "imageWidth")

    def __init__(self, fpath: str) -> None:
        """
        :param self: Description
        :param fpath: Description
        :type fpath: str
        """

        try:
            with open(fpath, mode="rt") as fp:
                _raw_ann = json.load(fp)
        except (FileNotFoundError, PermissionError) as excpt:
            raise RuntimeError(f"Failed to create an RLEAnnotation object from {fpath}") from excpt

        if not all([key in _raw_ann.keys() for key in RLEAnnotation.LABELME_ESSENTIAL_JSON_KEYS]):
            raise TypeError(f"Annotation JSON file must contain all of the following keys: {(*RLEAnnotation.LABELME_ESSENTIAL_JSON_KEYS,)}")

        # at this point, key subscripts should not raise an exception!!!
        self._fname = _raw_ann["imagePath"]
        self._height = _raw_ann["imageHeight"]
        self._width = _raw_ann["imageWidth"]
        self._nmasks = len(_raw_ann["shapes"])  # a single annotation file can contain multiple masks
        self._polygons = _raw_ann["shapes"]

    def shape(self) -> tuple[int, int]:
        """
        :return: Description
        :rtype: tuple[int, int]
        """
        return self._width, self._height

    def __len__(self) -> int:
        """
        :return: Description
        :rtype: int
        """
        return self._nmasks

    def to_coco_rle(self) -> dict[str, Any]:
        # the argument order for image_shape is (W, H)
        [rlemask.encode(polygon2mask(image_shape=(self._width, self._height), polygon=polygon["points"])) for polygon in self._polygons]

    def save(self, fpath: str) -> None:
        pass

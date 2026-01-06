# convert the raw JSON annotation files created by labelme to RLE compressed JSON files, which will be convenient when training on servers
# we do not want to upload huge JSON files, do we??? and it has the whole image file embedded in it???

import json
from typing import Any, Optional

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
        :param fpath: Description
        :type fpath: str
        """

        assert fpath.endswith(r".json"), "only .json files are accepted as annotations!"
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
        self._polygons = _raw_ann["shapes"]  # segmentation polygons
        # ["shapes"] returns a list of annotations, where each annotation is a dict with keys 'label', 'points', 'group_id', 'description', 'shape_type', 'flags'
        # "points" is a list of annotation coordinates (x, y) => list[[float, float]]

    def shape(self) -> tuple[int, int]:
        """
        :return: shape of the image (W, H) this object contains annotations for
        :rtype: tuple[int, int]
        """
        return self._width, self._height

    def __len__(self) -> int:
        """
        :return: number of annotations the object stores for the select image
        :rtype: int
        """
        return self._nmasks

    def to_coco_rle(self) -> dict[str, Any]:
        """
        :return: return a SA-1B style RLE'd convert of the original labelme annotations
        :rtype: dict[str, Any]

        lebelme stores segmentation annotations as polygons, defined by the (x, y) coordinates of the annotation points;
        >>> {
        'label': 'root',
        'points': [[248.51063829787222, 960.0000000000001],
        ([284.68085106382955, 923.8297872340427],)
        ...................
        ([276.17021276595733, 979.1489361702128],)
        [259.14893617021266, 987.6595744680851]],
        'group_id': 1,
        'description': '',
        'shape_type': 'polygon',
        'flags': {}
        }

        COCO RLE stores (as in the SA-1B dataset) them in the following way;
        >>> {
        'bbox': [694.0, 1379.0, 77.0, 128.0],
        'area': 9653,
        'segmentation': {'size': [1875, 1500],
        'counts': 'jShW1^1dh1P1kWNcMEa0jf1W3K3N1N100.....000001O000000000000001O0O2O1N5KX3fLhacY1'},
        'predicted_iou': 0.8956656455993652,
        'point_coords': [[728.3125, 1421.25]],
        'crop_box': [622.0, 1215.0, 567.0, 660.0],
        'id': 131671281,
        'stability_score': 0.9685015082359314
        }

        since this is purely for semantic segmentation, we can just focus on the "segmentation" key of the SA-1B annotation format
        the list of annotation coordinates can be conveniently converted to a 2D binary mask using skimage.draw.polygon2mask() function
        then this 2D binary mask can be RLE'd by pycoctools.mask.encode() to create the comprssed and compact run length encoding of the polygon!

        mask.deocde() expects a dict in the following format:
        >>> {
        'size': [W, H],
        'counts': '<RLE string>'
        }

        if we are really hard pressed to minimize the sizes of annotation files, we could avoid repeatedly storing the image dimensions in the "size" key and
        construct that in real time, using the object attributes.

        SA-1B annotation has 2 keys - "image" and "annotations"
        "image" is a dict with the following keys - 'image_id', 'width', 'height', 'file_name'
        and "annotations" is a list of dicts in the annotation format specified before.

        """

        return {
            "image": {
                "width": self._width,
                "height": self._height,
                "file_name": self._fname.replace(r".json", ""),  # file_name will only store the stem of the name without the extension
            },
            "annotations": [
                {
                    "label": polygon["label"],  # e.g. root or hyphae
                    "id": polygon["group_id"],  # preserving the "group_id" attr of labelme annotation in the "id" attr of SA-1B annotation
                    "counts": rlemask.encode(  # the argument order for image_shape in skimage.draw.polygon2mask() is (W, H)
                        polygon2mask(image_shape=(self._width, self._height), polygon=polygon["points"])
                    ),
                }
                for polygon in self._polygons
            ],
        }

        # since this "annotations" section does not have an explicit "segmentation" section, to deocde the RLE, we'd have to create a small dict, in real time
        # e.g. pycocotools.mask.decode({"size": [seg["image"]["width"], seg["image"]["height"]], "counts": seg["annotations"][0]["counts"]})

    def save_as_rle(self, fpath: Optional[str]) -> None:
        try:
            with open(file=fpath, mode="wt") as fp:  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]
                json.dump(obj=self.to_coco_rle(), fp=fp)  # pyright: ignore[reportUnknownArgumentType]
        except (FileNotFoundError, PermissionError) as excpt:
            raise RuntimeError("") from excpt

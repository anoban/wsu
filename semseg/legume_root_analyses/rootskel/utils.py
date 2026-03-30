import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

PIXELS_PER_MM: float = 45.9  # number of pixels in a millimeter
UCHAR_MIN: int = 0
UCHAR_MAX: int = 255


def open_and_skeletonize(fpath: str) -> NDArray[np.integer | np.floating]:
    """
    Read in the image file and apply the necessary transforms (skeletonization)
    Won't need this when we have image masks to work with (which is basically equivalent to the skeletonized output)
    Opting for inplace transformation of arrays, hoping we could squeeze out a bit more performance
    """

    with open(file=fpath, mode="rb") as fp:
        _image = Image.open(fp)
        if _image.mode != "RGB":
            raise TypeError(f"Only images with RGB colour channels are supported, got {_image.mode} for {fpath}!")
        _image = np.array(_image)

    _image = cv2.cvtColor(
        src=_image, code=cv2.COLOR_RGB2GRAY
    )  # input has three colour channels while the result only has one, cannot coerce the result into the input inplace
    cv2.bitwise_not(src=_image, dst=_image)  # inplace modification works here because the input and the result have the same shape
    cv2.medianBlur(src=_image, ksize=5, dst=_image)
    cv2.threshold(
        src=_image, thresh=UCHAR_MIN, maxval=UCHAR_MAX, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=_image
    )  # inplace greyscale to binary colour transformation
    # cv2.threshold(src=_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE, dst=_image) # is this necessary???
    cv2.ximgproc.thinning(  # https://docs.opencv.org/4.13.0/d9/d29/namespacecv_1_1ximgproc.html#aa244a73deb4e58ae70ee96afe9d2460b
        src=_image, dst=_image
    )  # applies a binary blob thinning operation (in place), to achieve a skeletization of the input image
    return _image

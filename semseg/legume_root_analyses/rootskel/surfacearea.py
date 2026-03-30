import cv2
import numpy as np
from numpy.typing import NDArray
from utils import PIXELS_PER_MM, UCHAR_MAX, UCHAR_MIN


def surface_area(_image: NDArray[np.integer | np.floating]) -> float:
    """ """

    assert len(_image.shape) == 2, "only single channel greyscale images are accepted!"

    cv2.bitwise_not(src=_image, dst=_image)
    cv2.medianBlur(src=_image, ksize=5, dst=_image)
    cv2.threshold(src=_image, thresh=UCHAR_MIN, maxval=UCHAR_MAX, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=_image)
    cv2.threshold(src=_image, thresh=UCHAR_MIN, maxval=UCHAR_MAX, type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE, dst=_image)
    cv2.ximgproc.thinning(src=_image, dst=_image)

    _distance = cv2.distanceTransform(src=_image, distanceType=cv2.DIST_L2, maskSize=5)
    surface_area = 0.0000

    for y in range(_image.shape[0]):
        for x in range(_image.shape[1]):
            if _image[y, x] > 0:
                radius = _distance[y, x] / PIXELS_PER_MM
                circumference = 2 * np.pi * radius
                surface_area += circumference

    return surface_area / PIXELS_PER_MM

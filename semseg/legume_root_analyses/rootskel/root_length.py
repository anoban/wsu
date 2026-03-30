# https://github.com/AG9843/Legume-Root-Analysis/blob/main/TRL.py

import cv2
import numpy as np
from numpy.typing import NDArray
from params import PIXEL_SIZE_CENTIMETERS

image = cv2.imread("D:/1.tiff")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
blur = cv2.medianBlur(image, 5)
ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
skeleton = cv2.ximgproc.thinning(image)


def component_length(skeletonized_component: NDArray[np.integer | np.floating], scale: bool = False) -> float:
    """
    For every connected component in the skeletonized image (can be imagined as fragments of roots or continuous root segments),
    calculate its length
    """

    coords = np.column_stack(np.where(skeletonized_component > 0))
    length = 0.00000
    for j in range(len(coords) - 1):
        dx = abs(coords[j + 1][0] - coords[j][0])
        dy = abs(coords[j + 1][1] - coords[j][1])
        if dx == dy == 1:
            length += 1.4142
        else:
            length += 1
    return length if not scale else length * PIXEL_SIZE_CENTIMETERS


def total_length(skeletonized_image: NDArray[np.integer | np.floating]) -> float:
    """
    Extract the connected components from the skeletonized image and calculate the cumulative length of the components
    The core functionality of this function is provided by the OpenCV function `connectedComponentsWithStats`.
    """

    # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(skeletonized_image)
    total_length = 0.00000
    noise_threshold = 1
    for i in range(1, num_components):
        if stats[i, cv2.CC_STAT_AREA] > noise_threshold:
            component = (labels == i).astype(np.uint8)
            total_length += component_length(component, scale=False)
    return total_length * PIXEL_SIZE_CENTIMETERS


total_length = total_length(skeleton)
print("TRL is", total_length * 0.0063)

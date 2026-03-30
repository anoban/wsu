# https://github.com/AG9843/Legume-Root-Analysis/blob/main/TRL.py

import cv2
import numpy as np
import utils
from numpy.typing import NDArray


def component_length(skeletonized_component: NDArray[np.integer | np.floating], scale: bool = False) -> float:
    """
    For every connected component in the skeletonized image (can be imagined as fragments of roots or continuous root segments),
    calculate its length
    With `scale` argument set to True, the unit of the returned length will be in millimeters, else just pixel counts (scale=False, the default)
    """

    coords = np.column_stack(np.where(skeletonized_component > 0))
    length = 0.00000
    for j in range(len(coords) - 1):
        dx = abs(coords[j + 1][0] - coords[j][0])
        dy = abs(coords[j + 1][1] - coords[j][1])
        raise NotImplementedError("Incomplete implementation!")
        if dx == dy == 1:
            length += None  # what's this number????
        else:
            length += None  # this too????
    return length if not scale else (length / utils.PIXELS_PER_MM)


def total_length(skeletonized_image: NDArray[np.integer | np.floating]) -> float:
    """
    Extract the connected components from the skeletonized image and calculate the cumulative length of the components
    The core functionality of this function is provided by the OpenCV function `connectedComponentsWithStats`.
    The unit of the returned length is in millimeters
    """

    # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(skeletonized_image)
    total_length = 0.00000
    noise_threshold = 1
    for i in range(1, nlabels):
        if stats[i, cv2.CC_STAT_AREA] > noise_threshold:
            component = (labels == i).astype(np.uint8)
            total_length += component_length(component, scale=False)
    return total_length / utils.PIXELS_PER_MM

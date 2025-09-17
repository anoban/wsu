import numpy as np
from numpy.typing import NDArray

from .params import PIXEL_SIZE_CENTIMETERS

# We'll port this to use scikitimage instead of the bloated OpenCV

image = cv2.imread("D:/1.tiff")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
blur = cv2.medianBlur(image, 5)
ret3, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
skeleton_image = cv2.ximgproc.thinning(image)


num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton_image)
length = stats[:, cv2.CC_STAT_AREA]
length = list(sorted(length))
assert len(length) >= 2

total_length = 0
noise = 250
for i, length in enumerate(length[:-1]):
    if length > noise:
        total_length += length
        TRL = total_length

distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
diameter_sum = 0

for y in range(skeleton_image.shape[0]):
    for x in range(skeleton_image.shape[1]):
        if skeleton_image[y, x] > 0:  # Check if the pixel is part of the skeleton
            radius = distance_transform[y, x] * PIXEL_SIZE_CENTIMETERS
            diameter = 2 * radius
            diameter_sum += diameter
average_diameter = diameter_sum / TRL

print("Average Diameter (cm):", average_diameter)


def __average_diameter(skeletonized_image: NDArray[np.floating | np.integer]) -> float:
    """ """

    for y in range(skeleton_image.shape[0]):
        for x in range(skeleton_image.shape[1]):
            if skeleton_image[y, x] > 0:  # Check if the pixel is part of the skeleton
                radius = distance_transform[y, x] * PIXEL_SIZE_CENTIMETERS
                diameter = 2 * radius
                diameter_sum += diameter
    return diameter_sum / TRL

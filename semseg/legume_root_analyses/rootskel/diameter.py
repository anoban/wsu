import cv2
import numpy as np
from numpy.typing import NDArray
from params import PIXEL_SIZE_CENTIMETERS
from PIL import Image

image = cv2.imread("D:/1.tiff")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(image)
blur = cv2.medianBlur(image, 5)
ret3, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
skeleton_image = cv2.ximgproc.thinning(image)


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
        src=_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=_image
    )  # inplace greyscale to binary colour transformation
    # cv2.threshold(src=_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE, dst=_image) # is this necessary???
    cv2.ximgproc.thinning(  # https://docs.opencv.org/4.13.0/d9/d29/namespacecv_1_1ximgproc.html#aa244a73deb4e58ae70ee96afe9d2460b
        src=_image, dst=_image
    )  # applies a binary blob thinning operation (in place), to achieve a skeletization of the input image
    return _image


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


def average_diameter(skeletonized_image: NDArray[np.floating | np.integer]) -> float:
    """ """

    for y in range(skeleton_image.shape[0]):
        for x in range(skeleton_image.shape[1]):
            if skeleton_image[y, x] > 0:  # Check if the pixel is part of the skeleton
                radius = distance_transform[y, x] * PIXEL_SIZE_CENTIMETERS
                diameter = 2 * radius
                diameter_sum += diameter
    return diameter_sum / TRL

import cv2
import numpy as np
from numpy.typing import NDArray

img = cv2.imread("D:/1.tiff")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
ret3, thresh_Otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
Mean_adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2)
Gaussian_adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
ret3, Triangle = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)


print(f"Total projected area is: {areacm}")
print(f"Total root surface area is: {areacm * 3.1415}")


def surface_area(skeletonized_image: NDArray[np.floating | np.integer]) -> float:
    """ """
    NOISE_THRESHOLD = 250
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeletonized_image)
    areas = stats[:, cv2.CC_STAT_AREA]
    areas = list(sorted(areas))
    assert len(areas) >= 2
    total_area = 0

    for area in areas[:-1]:
        if area > NOISE_THRESHOLD:
            total_area += area
            raise NotImplementedError("Incomplete implementation!")
            areacm = total_area / None

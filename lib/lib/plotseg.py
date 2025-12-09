import cv2
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

__all__: list[str] = ["plot_segmentations"]


def plot_segmentations(ax: Axes, masks: NDArray[np.integer], height: int, width: int, alpha: float = 0.35, borders: bool = True) -> Axes:
    """
    overlay segmentation results of a segmentation model on an image (or empty axes)
    the parameter `masks` is expected to be a 3 dimensional numpy array of shape (N, H, W) where N is the number of binary masks,
    H is expected to match the height of the image and W is expected to match the width of the image.
    each mask in the masks object is expected to be transformable to a boolean matrix, to subset the pixels that need colouring during plotting.
    alpha is transparency applied to the mask when it's overlaid on the image, for better transparency, use higher alpha values.
    borders is to specify whether a border needs to be drawn, outlining each segmentation result, instead of using colours alone to delineate the objects.
    """

    assert masks.shape[1] == height and masks.shape[2] == width, "dimensions of the masks do not match the params height and width"

    rgba = np.ones((height, width, 4))
    for mask in masks:
        rgba[:, :, 3] = 0  # set all the alpha channel values to 0
        # use the boolean mask to cherry pick the elements where the mask applies
        rgba[mask.astype(np.bool), :] = np.array(
            [*np.random.random(3), alpha]
        )  # and update the colour (RGB) values with the specified alpha value
        ax.imshow(rgba)  # type: ignore

        if borders:  # this block of code is adapted from https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # type: ignore
            # try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]  # type: ignore
            cv2.drawContours(image=rgba, contours=contours, contourIdx=-1, color=(0, 0, 1, alpha), thickness=1)

    return ax

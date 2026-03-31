import cv2
import torch
import numpy as np
from typing import Any
from matplotlib.axes import Axes
from numpy.typing import NDArray

__all__: list[str] = ["rgb_image_to_tensor", "tensor_to_rgb_image", "downscale_to_uchars", "plot_segmentations"]


def rgb_image_to_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    """
    input is expected to be a H x W matrix made of [R, G, B] channels
    i.e (H, W, 3) to (3, H, W)
    """

    height, width, nchannels = image.shape
    if nchannels != 3:  # RGB channels expected!!!
        raise RuntimeError(f"Only images using the standard RGB channel pixels are supported! Expected 3, but got {nchannels} channels!")

    result = torch.tensor(
        np.array(np.unstack(image, axis=-1)),  # split the image into 3 matrixes for each R, G and B channels
        dtype=torch.float32,
    )

    assert (result.shape[0] == 3) and (result.shape[1] == height) and (result.shape[2] == width), (
        "Shape mismatches between input array and result tensor!"
    )
    return result


def tensor_to_rgb_image(tensor: torch.Tensor) -> NDArray[np.float32]:
    """
    transforms a tensor of R, G & B matrices into a matrix of [R, G, B] pixels
    i.e from (3, H, W) to (H, W, 3)
    """

    return np.stack(tensor.numpy(force=False), axis=-1, dtype=np.float32)  # type: ignore


@torch.no_grad()  # type: ignore
def downscale_to_uchars(tensor: torch.Tensor | NDArray[np.floating]) -> torch.Tensor | NDArray[np.uint8]:
    """
    scales a tensor or array of floats with unknown bounds into a tensor or array of uint8s with an inclusive range [0, 255]
    """

    tensor += abs(tensor.min())  # probably a negative value, hence the abs()
    tensor /= tensor.max()  # downscale to [0.00, 1.00] (inclusive range)
    tensor *= 255  # upscale to RGB channel max 255
    return tensor.type(torch.uint8) if isinstance(tensor, torch.Tensor) else tensor.astype(np.uint8)


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


def plot_predictions(segmentations: list[dict[str, Any]], axes: Axes) -> None:
    """
    an alternative to the plot_segmentations function, that's designed to handle raw outputs from SAM and SAM2 models.
    this output is expected to be a list of segmentation dictoionaries
    """
    
    if len(segmentations) == 0: # if the segmentation list is empty
        raise ValueError("Argument 'segmentations' cannot be empty!")
    
    sorted_segmentations = sorted(segmentations, key=(lambda x: x['area']), reverse=True) # sort the masks in ascending order of area 
    axes.set_autoscale_on(False)
    # create a rgba colour channel tensor with width and height matching the segmentation masks
    rgba = np.ones((sorted_segmentations[0]['segmentation'].shape[0], # width
                    sorted_segmentations[0]['segmentation'].shape[1], # height
                    4)) 
    rgba[:, :, 3] = 0 # set all the alpha channel values to 0
    for seg in sorted_segmentations:
        mask = seg['segmentation'].astype(np.bool) # capture the boolean mask matrix
        rgba[mask] = np.concatenate([np.random.random(3), [0.35]]) # update the masked region with a new RGBA colour - random RGB with a fixed alpha value
    axes.imshow(rgba)
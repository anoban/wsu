import numpy as np
import torch
from numpy.typing import NDArray


def rgb_matrix_to_tensor(image: NDArray[np.uint8]) -> torch.FloatTensor:
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


def tensor_to_rgb_matrix(tensor: torch.FloatTensor) -> torch.FloatTensor:
    """
    transforms a tensor of R, G & B matrices into a matrix of [R, G, B] pixels
    i.e from (3, H, W) to (H, W, 3)
    """

    return np.stack(tensor, axis=-1)


@torch.no_grad
def scale_to_standard_rgb_channels(tensor: torch.FloatTensor | NDArray[np.floating]) -> torch.IntTensor | NDArray[np.uint8]:
    """
    scales a tensor or array of floats with unknown bounds into a tensor or array of uint8s with an inclusive range [0, 255]
    """

    tensor += abs(tensor.min())  # probably a negative value, hence the abs()
    tensor /= tensor.max()  # downscale to [0.00, 1.00]
    tensor *= 255  # upscale to RGB channel max 255
    return tensor.type(torch.uint8) if isinstance(tensor, torch.Tensor) else tensor.astype(np.uint8)

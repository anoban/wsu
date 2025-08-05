import numpy as np
import torch
from numpy.typing import NDArray

__all__: list[str] = ["rgb_image_to_tensor", "tensor_to_rgb_image", "downscale_to_uchars"]


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

    return np.stack(tensor, axis=-1, dtype=np.float32)


@torch.no_grad  # type: ignore
def downscale_to_uchars(tensor: torch.Tensor | NDArray[np.floating]) -> torch.Tensor | NDArray[np.uint8]:
    """
    scales a tensor or array of floats with unknown bounds into a tensor or array of uint8s with an inclusive range [0, 255]
    """

    tensor += abs(tensor.min())  # probably a negative value, hence the abs()
    tensor /= tensor.max()  # downscale to [0.00, 1.00] (inclusive range)
    tensor *= 255  # upscale to RGB channel max 255
    return tensor.type(torch.uint8) if isinstance(tensor, torch.Tensor) else tensor.astype(np.uint8)

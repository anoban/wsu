from typing import Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class RootSkeleton(object):
    """
    #
    """

    def __init__(self, filepath_or_image: Union[str, Image.Image, NDArray[np.floating | np.integer]], colour_channel: str | None) -> None:
        """
        #
        """

        if isinstance(filepath_or_image, str):  # if `image` is a file path
            try:
                with open(file=filepath_or_image, mode="rb") as fp:
                    image = Image.open(fp=fp)
                    self.colourchannel = image.mode
                    self.pixels = np.array(image, dtype=np.float32)
            except IOError as ioexcept:
                raise RuntimeError("") from ioexcept  # TODO
        elif isinstance(filepath_or_image, Image.Image):  # type: ignore
            self.pixels = np.array(filepath_or_image, dtype=np.float32)
            self.colourchannel = filepath_or_image.mode
        elif isinstance(filepath_or_image, np.ndarray):  # type: ignore
            if not colour_channel:
                raise ValueError(
                    r"When the input is a Numpy array, argument colour_channel must be explicitly specified as it cannot be inferred!"
                )
            self.pixels = filepath_or_image
            self.colourchannel = colour_channel
        else:
            raise TypeError(r"Only strings, PIL.Image objects and Numpy arrays are accepted as inputs!")

        # array.shape() will return (h, w, nch) when there are more than one colour channles and (h, w) when the image is a single channel image
        self.height, self.width, self.n_colourchannels = self.pixels.shape if len(self.pixels.shape) == 3 else (*self.pixels.shape, 1)

        # handle the situation when the image has an alpha channel

    @staticmethod
    def _impl_skeletonize(image: NDArray[np.floating | np.integer]) -> None:
        """
        Steps::
            1. convert the pixels to single channel greyscale representation
            2.
        """

        # implement what's needed to reproduce the following with skimage
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) - use skimage.color.rgb2gray()
        # colour channels seem to be reversed through!!!!!!
        # Y = 0.2125 R + 0.7154 G + 0.0721 B

        # image = cv2.bitwise_not(image)
        # blur = cv2.medianBlur(image, 5)
        # ret3, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        # skeleton_image = cv2.ximgproc.thinning(image)

    def volume(self) -> float:
        """ """
        return 0.000

    def surface_area(self) -> float:
        """ """
        return 0.000

    def total_length(self) -> float:
        """ """
        return 0.000

    def average_diameter(self) -> float:
        """ """
        return 0.000

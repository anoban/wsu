import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Skeleton(object):
    """
    #
    """

    def __init__(self, image: str | Image.Image) -> None:
        if isinstance(image, str):  # if `image` is a file path
            try:
                with open(file=image, mode="rb") as fp:
                    self.pixels = np.array(Image.open(fp=fp), dtype=np.float32)
                    self.width = 0  # TODO
                    self.height = 0  # TODO
                    self.colour_channels = 3  # TODO
            except IOError as ioexcept:
                raise RuntimeError("") from ioexcept  # TODO
        elif isinstance(image, Image.Image):
            self.pixels = np.array(image, dtype=np.float32)
            self.width = 0  # TODO
            self.height = 0  # TODO
            self.colour_channels = 3  # TODO
        else:
            raise TypeError("@@@@")  # TODO

    @staticmethod
    def _impl_skeletonize(image: NDArray[np.floating | np.integer] | Image.Image) -> None:
        """
        #
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

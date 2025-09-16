from typing import Self

import numpy as np
from PIL import Image


class skeleton(object):
    """
    #
    """

    def __init__(self, filepath: str) -> None:
        try:
            with open(file=filepath, mode="rb") as fp:
                self.pixels = np.array(Image.open(fp=fp), dtype=np.float32)
                self.width = 0
                self.height = 0
                self.colour_channels = 3
        except IOError as ioexcept:
            raise RuntimeError("") from ioexcept

    def skeletonize(self, inplace: bool = False) -> Self | None:
        """
        #
        """

        pass

    def volume(self) -> float:
        """ """
        return 0.000

    def surface_area(self) -> float:
        """ """
        return 0.000

    def total_length(self) -> float:
        """ """
        return 0.000

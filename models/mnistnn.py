import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax

__doc__ = r"A collection of CNN classifiers with different architecures for MNIST style datasets"
__all__ = (r"BareBonesNN",)


class BareBonesNN(nn.Module):
    """
    Architecture:

    input layer   - 01
    hidden layers - 01
    output layer  - 01

    """

    def __init__(self, n_classes: int = 10) -> None:
        """ """

        super(BareBonesNN, self).__init__()  # type: ignore

        self._fucon_01 = nn.Linear(
            in_features=784,  # 28 x 28 pixels
            out_features=784 * 2,
        )
        self._fucon_02 = nn.Linear(in_features=784 * 3, out_features=n_classes)

    def forwrad(self, image: torch.Tensor) -> torch.Tensor:
        image = self._fucon_01(image)
        image = relu(image)
        image = self._fucon_02(image)
        return softmax(image)

    def learn(self) -> None:
        super().train(mode=True)

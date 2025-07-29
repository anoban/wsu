import torch
import torch.nn as nn
from torch.nn.functional import nll_loss, relu, softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..lib import IdxDataset

__doc__ = r"A collection of NN classifiers with different architecures for MNIST style datasets"
__all__ = (r"BareBonesNN",)


class BareBonesNN(nn.Module):
    """
    a barebones neural network that doesn't use convolutional layers

    Architecture::
    input layer   - 01
    hidden layers - 01
    output layer  - 01

    """

    def __init__(self, n_classes: int = 10) -> None:
        """ """

        super(BareBonesNN, self).__init__()  # type: ignore

        self._fucon_01 = nn.Linear(
            in_features=784,  # 28 x 28 pixels
            out_features=784 * 2,  # this is an arbitrary choice
        )
        self._fucon_02 = nn.Linear(
            in_features=784 * 2,  # same as the output features of the previous layer
            out_features=n_classes,
        )

    def forwrad(self, image: torch.Tensor) -> torch.Tensor:
        image = self._fucon_01(image)
        image = relu(image)
        image = self._fucon_02(image)
        return softmax(image, dim=1)

    def learn(self, path_images: str, path_labels: str, optimizer: Optimizer, n_epochs: int = 100) -> None:
        super().train(mode=True)
        train_loader = DataLoader(
            dataset=IdxDataset(idx3_filepath=path_images, idx1_filepath=path_labels), batch_size=1, shuffle=True, num_workers=6
        )

        for _ in range(n_epochs):
            for image, label in train_loader:
                optimizer.zero_grad()
                result = self.forwrad(image=image)
                loss = nll_loss(input=result, target=label)
                loss.backward()  # type: ignore
                optimizer.step()

    def save(self, path_without_extension: str) -> None:
        """ """

        with open(file=f"{path_without_extension}.trch", mode="rb") as fp:
            # PyTorch recommends .pt or .pth extensions but .trch sounds way cooler :)
            torch.save(obj=self, f=fp)

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        super().eval()  # equivalent to super().train(mode=False)
        return self.forward(image).argmax()

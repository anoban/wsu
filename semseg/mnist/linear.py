from typing import override
from warnings import warn

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, relu
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class LiNN(nn.Module):
    """
    A fully connected linear neural network without any convolutions
    """

    def __init__(self) -> None:
        """ """

        super(LiNN, self).__init__()  # type: ignore

        self.fconn_01 = nn.Linear(
            in_features=784,  # 28x28 pixels of the flattened image
            out_features=1024,
        )
        self.fconn_02 = nn.Linear(in_features=1024, out_features=1448)
        self.fconn_03 = nn.Linear(in_features=1448, out_features=512)
        self.dropout = nn.Dropout(p=0.25)
        self.fconn_04 = nn.Linear(in_features=512, out_features=128)
        self.fconn_05 = nn.Linear(in_features=128, out_features=10)

    @override
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """ """

        super().train(mode=True)

        batch = self.fconn_01(torch.flatten(batch, start_dim=1))
        batch = relu(batch)

        batch = self.fconn_02(batch)
        batch = relu(batch)

        batch = self.fconn_03(batch)
        batch = relu(batch)
        batch = self.dropout(batch)

        batch = self.fconn_04(batch)
        batch = relu(batch)

        batch = self.fconn_05(batch)
        probs = log_softmax(batch, dim=1)

        return probs

    def fit(
        self,
        train_loader: DataLoader[torch.Tensor],
        test_loader: DataLoader[torch.Tensor],
        optimizer: Optimizer,
        device: torch.device = torch.device("cpu", 0),
        gamma: float = 0.7,
        epochs: int = 20,
        log_interval: int = 10,
    ) -> None:
        super().train(mode=True)
        super().to(device=device)

    @torch.no_grad()  # type: ignore
    def _impl_evaluate(self, test_loader: DataLoader[torch.Tensor], device: torch.device = torch.device("cpu", 0)) -> None:
        """
        Evaluate the model's current state's performance on the test dataset
        This method is not mean to be directly invoked by users
        """

        super().eval()
        loss: float = 0.000
        correct_predictions: int = 0

        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)  # move the pair of tensors to the specified device
            probs = self(batch)  # predicted probabilities for labels

    def to_disk(self, path: str) -> None:
        if not path.endswith(r".pt") and not path.endswith(r".pt"):  # using .pt or .pth extensions is recommended
            warn(r"It's advised to use .pt or .pth extensions when serializing PyTorch models!")
        try:
            with open(file=path, mode=r"wb") as fp:
                torch.save(obj=super().state_dict(), f=fp)
        except IOError as ioexcept:
            raise RuntimeError(f"Cannot open file {path} for writing because of {ioexcept.strerror}")


def main() -> None:
    train = MNIST(root=r"../../data/", train=True, download=False)
    test = MNIST(root=r"../../data/", train=False, download=False)

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = LiNN()

    optimizer = SGD(params=model.parameters(), lr=0.9, momentum=0.9)
    criterion = nn.CrossEntropyLoss()


if __name__ == r"__main__":
    main()

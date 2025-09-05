from typing import override

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, relu
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader


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
    def evaluate(self, test_loader: DataLoader[torch.Tensor], device: torch.device = torch.device("cpu", 0)) -> None:
        super().eval()

    def to_disk(self, path: str) -> None:
        pass


def main() -> None:
    train = IdxDataset(r"./FashionMNIST/train-labels-idx1-ubyte", r"./FashionMNIST/train-images-idx3-ubyte")
    test = IdxDataset(r"./FashionMNIST/t10k-labels-idx1-ubyte", r"./FashionMNIST/t10k-images-idx3-ubyte")

    train_loader = DataLoader(dataset=train, batch_size=1, shuffle=True, num_workers=6)
    test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=6)

    model = LiNN()

    optimizer = SGD(params=model.parameters(), lr=0.001, momentum=0.900)
    criterion = nn.CrossEntropyLoss()




if __name__ == r"__main__":
    main()

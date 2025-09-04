# https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/mnist/main.py

import argparse
from typing import override
from warnings import warn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax, max_pool2d, nll_loss, relu
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from torchvision.datasets import MNIST  # type: ignore


class CNNet(nn.Module):
    def __init__(self):
        """
        Initialize the class instance and configure the layer layouts and the architecure
        """

        super(CNNet, self).__init__()  # type: ignore
        self.cnvltn_01 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=3)
        self.cnvltn_02 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=3)
        self.dropout_01 = nn.Dropout(0.25)
        self.dropout_02 = nn.Dropout(0.5)
        self.fully_cnctd_01 = nn.Linear(in_features=9216, out_features=128)
        self.fully_cnctd_02 = nn.Linear(in_features=128, out_features=10)

    @override
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        The method for performing a forward pass
        """

        batch = self.cnvltn_01(batch)
        batch = relu(batch)
        batch = self.cnvltn_02(batch)
        batch = relu(batch)
        batch = max_pool2d(batch, 2)
        batch = self.dropout_01(batch)
        batch = torch.flatten(batch, 1)
        batch = self.fully_cnctd_01(batch)
        batch = relu(batch)
        batch = self.dropout_02(batch)
        batch = self.fully_cnctd_02(batch)  # output features = 10
        output = log_softmax(batch, dim=1)
        return output

    def fit(
        self,
        args: argparse.Namespace,
        train_loader: DataLoader[torch.Tensor],
        optimizer: Optimizer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Fit the model to the training dataset
        """

        super().train(mode=True)  # set nn.Module parent class's state to training
        super().to(device=device)  # move the model to the specified device

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            # train(args, model, device, train_loader, optimizer, epoch)
            # test(model, device, test_loader)

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = nll_loss(output, target)
                loss.backward()  # type: ignore
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()
                        )
                    )

            if args.dry_run:  # if only a dry run, break after a single pass
                break

            scheduler.step()

    @torch.no_grad()  # type: ignore
    def evaluate(self, device: torch.device, test_loader: DataLoader[torch.Tensor]) -> int:
        """ """
        super().eval()  # set the state of parent class nn.Module to predictions, equivalent to self.train(mode=False)
        test_loss: float = 0.000
        correct: int = 0

        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            output = self(batch)  # overloaded __call__() of nn.Module invokes forward() and other necessary internal hooks under the hood
            test_loss += nll_loss(output, labels, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(
            test_loader.dataset  # type: ignore
        )  # calling the __len__() of DataLoader class can give misleading results as it provides the number of batches NOT the number of images in the Dataset
        # hence the need to call the __len__() of the Dataset class directly
        # since class Dataset does not have a default __len__() method defined, linters will bitch about the missing method
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader), 100.0 * correct / len(test_loader)
            )
        )

    def serialize(self, path: str) -> None:
        """
        Serialize the model's current state to a binary dictionary object
        """

        if not path.endswith(r".pt") and not path.endswith(r".pt"):  # using .pt or .pth extensions is recommended
            warn(r"It's advised to use .pt or .pth extensions when serializing PyTorch models!")
        try:
            with open(file=path, mode=r"wb") as fp:
                torch.save(super().state_dict(), f=fp)
        except IOError as ioexcpt:
            raise RuntimeError(f"Unable to serialize the model to file {path} because of exception {ioexcpt.strerror}")


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N", help="input batch size for testing (default: 64)")
    parser.add_argument("--epochs", type=int, default=1000, metavar="N", help="number of epochs to train (default: 1000)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument("--dry-run", action="store_true", help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    # parser.add_argument("--save-model", action="store_true", help="For Saving the current Model")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu", 0)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_accel:
        accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True, "shuffle": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )  # this is where the 2D PIL images get transformed and normalized into torch Tensors
    dataset1 = MNIST(r"../data/MNIST/", train=True, download=False, transform=transform)
    dataset2 = MNIST(r"../data/MNIST/", train=False, download=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = CNNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()


if __name__ == "__main__":
    main()

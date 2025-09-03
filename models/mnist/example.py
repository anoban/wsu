# https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/mnist/main.py

import argparse
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # type: ignore
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    @override
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.conv1(batch)
        batch = F.relu(batch)
        batch = self.conv2(batch)
        batch = F.relu(batch)
        batch = F.max_pool2d(batch, 2)
        batch = self.dropout1(batch)
        batch = torch.flatten(batch, 1)
        batch = self.fc1(batch)
        batch = F.relu(batch)
        batch = self.dropout2(batch)
        batch = self.fc2(batch)
        output = F.log_softmax(batch, dim=1)
        return output

    def fit(
        self, args, train_loader: DataLoader[torch.Tensor], optimizer: Optimizer, epoch: int, device: torch.device = torch.device("cpu")
    ) -> None:
        super().train(mode=True)  # set nn.Module parent class's state to training
        super().to(device=device)  # move the model to the specified device
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()  # type: ignore
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()
                    )
                )
                if args.dry_run:
                    break

    def predict(self, device: torch.device, test_loader: DataLoader[torch.Tensor]) -> int:
        super().eval()  # set the state of parent class nn.Module to predictions, equivalent to self.train(mode=False)
        test_loss: float = 0.000
        correct: int = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)  # calls forward() and other internal hooks under the hood
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

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
        """ """

        try:
            with open(file=path, mode=r"wb") as fp:
                torch.save(super().state_dict(), f=fp)
        except IOError as ioexcpt:
            raise RuntimeError("###############################") from ioexcpt


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument("--dry-run", action="store_true", help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", help="For Saving the current Model")
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_accel:
        accel_kwargs = {"num_workers": 1, "persistent_workers": True, "pin_memory": True, "shuffle": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()

import torch
from segment_anything.modeling.sam import Sam
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def custom_dataset_finetune(
    model: Sam, dtloader: DataLoader[torch.Tensor], optimizer: Optimizer, loss_fn: _Loss, lrate: float, n_epochs: int, log_intrvl: int
) -> None:
    """
    :param model: Description
    :type model: Sam
    :param dtloader: Description
    :type dtloader: DataLoader[torch.Tensor]
    :param optimizer: Description
    :type optimizer: Optimizer
    :param loss_fn: Description
    :param lrate: Description
    :type lrate: float
    :param n_epochs: Description
    :type n_epochs: int
    :param log_intrvl: Description
    :type log_intrvl: int
    """

    pass

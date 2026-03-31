import torch
from segment_anything.modeling.sam import Sam
from torch.nn.modules.loss import _Loss  # pyright: ignore[reportPrivateUsage]
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# for a discourse on object overlap in segmentation annotations, have a look at https://github.com/ultralytics/ultralytics/issues/3213


def custom_dataset_finetune(
    model: Sam, dtloader: DataLoader[torch.Tensor], optimizer: Optimizer, loss_fn: _Loss, lrate: float, n_epochs: int, log_intrvl: int
) -> None:
    """
    :param model: pretrained Sam model object
    :type model: Sam
    :param dtloader: data loader (images and annotations)
    :type dtloader: DataLoader[torch.Tensor]
    :param optimizer: optimizer
    :type optimizer: Optimizer
    :param loss_fn: loss function to estimate the prediction accuracy
    :param lrate: Description
    :type lrate: learning rate
    :param n_epochs: number of iterations to repeat the training
    :type n_epochs: int
    :param log_intrvl: how often you want the model's performance to be logged to the stdout
    :type log_intrvl: int
    """

    pass

import torch
from poutyne.framework.metrics import acc


def accuracy(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Accuracy per tag.
    """
    return acc(pred.transpose(0, 1).transpose(-1, 1), ground_truth)

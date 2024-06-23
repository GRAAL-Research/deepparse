import torch
from poutyne.framework.metrics import acc


def accuracy(predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
    """
    Accuracy per tag.
    """
    return acc(predictions.transpose(0, 1).transpose(-1, 1), ground_truths)

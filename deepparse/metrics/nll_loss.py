import torch
from torch.nn import NLLLoss

criterion = NLLLoss()


def nll_loss(predictions: torch.Tensor, ground_truths: torch.Tensor) -> float:
    """
    NLL loss to compute loss per tag.
    """
    loss = 0

    ground_truths = ground_truths.transpose(0, 1)
    for i in range(predictions.size(0)):
        loss += criterion(predictions[i], ground_truths[i])
    return loss

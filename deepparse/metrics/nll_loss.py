import torch
from torch.nn import NLLLoss

criterion = NLLLoss()


def nll_loss(pred: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    NLL loss compute per tag.
    """
    loss = 0

    ground_truth = ground_truth.transpose(0, 1)
    for i in range(pred.size(0)):
        loss += criterion(pred[i], ground_truth[i])
    return loss

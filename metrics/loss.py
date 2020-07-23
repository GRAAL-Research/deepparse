from torch.nn import NLLLoss


def nll_loss_function(pred, ground_truth):
    criterion = NLLLoss()
    loss = 0

    ground_truth = ground_truth.transpose(0, 1)
    for i in range(pred.size(0)):
        loss += criterion(pred[i], ground_truth[i])

    return loss

from torch.nn import NLLLoss


def nll_loss_function(pred, ground_truth):
    criterion = NLLLoss()

    loss = 0

    for i in range(pred.size(0)):
        loss += criterion(pred[i].view(1, 9), ground_truth[0][i].view(1))

    return loss

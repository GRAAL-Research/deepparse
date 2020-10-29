from poutyne.framework.metrics import acc


def accuracy(pred, ground_truth):
    return acc(pred, ground_truth[0])

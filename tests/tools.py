import torch

from deepparse.dataset_container import DatasetContainer


def create_file(path: str, content: str):
    """"
    Function to create a file for test
    """
    with open(path, "w") as f:
        f.write(content)


BATCH_SIZE = 32


class ADataContainer(DatasetContainer):

    def __init__(self, ):
        super().__init__()
        self.data = (torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))

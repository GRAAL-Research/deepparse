from abc import ABC, abstractmethod
from pickle import load

from torch.utils.data import Dataset


class DatasetContainerInterface(Dataset, ABC):
    """
    Interface for the dataset. This interface define most of the method that the dataset need to define.
    If you define other dataset container, the init must define the attribute data.
    """

    @abstractmethod
    def __init__(self):
        """
        Need to be define by child class.
        """
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            result = []
            for element in range(idx.start, idx.stop):
                sample = self.data[element]

                result.append(sample)
        else:
            result = self.data[idx]

        return result


class PickleDatasetContainer(DatasetContainerInterface):
    """
    Pickle dataset container that import a list of address in pickle format.
    See `here <https://github.com/GRAAL-Research/poutyne-external-assets/tree/master/tips_and_tricks_assets>`_ for
    example of pickle address data.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data = load(open(data_path, 'rb'))

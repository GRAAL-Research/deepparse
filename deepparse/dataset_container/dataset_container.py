from abc import ABC, abstractmethod
from pickle import load
from typing import Union

from torch.utils.data import Dataset


class DatasetContainer(Dataset, ABC):
    """
    Interface for the dataset. This interface define most of the method that the dataset needs to define.
    If you define another dataset container, the init must define the attribute data.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Need to be define by child class.
        """
        self.data = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            result = []
            for element in range(idx.start, idx.stop):
                sample = self.data[element]

                result.append(sample)
        else:
            result = self.data[idx]

        return result


class PickleDatasetContainer(DatasetContainer):
    """
    Pickle dataset container that import a list of address in pickle format.
    See `here <https://github.com/GRAAL-Research/poutyne-external-assets/tree/master/tips_and_tricks_assets>`_ for
    example of pickle address data.
    """

    def __init__(self, data_path: str) -> None:
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data = load(f)

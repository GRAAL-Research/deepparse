from torch.utils.data import Dataset
from pickle import load

class DatasetContainer(Dataset):
    def __init__(self, data_path, transform=None):
       self.data = load(open(data_path, 'rb'))
       self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            result = []
            for element in range(idx.start, idx.stop):
                sample = self.data[element]

                if self.transform:
                    sample = self.transform(sample)
                result.append(sample)
        else:
            result = self.data[idx]

            if self.transform:
                result = self.transform(result)

        return result


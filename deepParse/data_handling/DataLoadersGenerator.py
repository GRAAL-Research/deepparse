from torch.utils.data import DataLoader


class DataLoadersGenerator:
    def __init__(self, dataset, train_ratio, teacher_forcing_collate, output_reuse_collate, batch_size, num_workers):
        self.dataset = dataset
        self.size = len(dataset)
        self.train_ratio = train_ratio
        self.teacher_forcing_collate = teacher_forcing_collate
        self.output_reuse_collate = output_reuse_collate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def generate_dataloaders(self):
        train_dataset = []
        for pair in self.dataset[0:int(self.size * self.train_ratio)]:
            train_dataset.append((pair[0], pair[1]))

        train_generator = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, collate_fn=self.teacher_forcing_collate, num_workers=self.num_workers)

        valid_dataset  = []
        for pair in self.dataset[int(self.size * self.train_ratio):self.size]:
            valid_dataset.append((pair[0], pair[1]))

        valid_generator = DataLoader(valid_dataset, batch_size=self.batch_size, drop_last=True, collate_fn=self.output_reuse_collate, num_workers=self.num_workers)

        return train_generator, valid_generator 
        
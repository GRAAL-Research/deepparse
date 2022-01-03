# A class to mock to logic of using a collate_fn in the data loader to be multiprocess and test if multiprocess work
class MockedDataTransform:
    def __init__(self, word_vectors_model):
        self.model = word_vectors_model

    def collate_fn(self, x):
        words = []
        for data_sample in x:
            for word in data_sample[0].split():
                words.append(self.model(word))
        return words

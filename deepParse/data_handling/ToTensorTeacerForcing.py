from deepParse.data_handling.ToTensor import ToTensor

class ToTensorTeacerForcing(ToTensor):
    def __init__(self, embedding_size, vectorizer, padding_value, device, mask_value=-100):
        super().__init__(embedding_size, vectorizer, padding_value, device, mask_value=-100)

    def _teacher_forcing_transform(self, pairs_batch):
        transformed_batch = super()._transform(pairs_batch)

        return (transformed_batch, transformed_batch[-1])

    def transform_function(self):
        return self._teacher_forcing_transform
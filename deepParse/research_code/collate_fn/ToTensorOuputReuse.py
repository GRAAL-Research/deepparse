from deepParse.research_code.collate_fn.ToTensor import ToTensor


class ToTensorOuputReuse(ToTensor):
    def __init__(self, embedding_size, vectorizer, padding_value, device, mask_value=-100):
        super().__init__(embedding_size, vectorizer, padding_value, device, mask_value=-100)

    def _ouput_reuse_transform(self, pairs_batch):
        transformed_batch = super()._transform(pairs_batch)

        return (transformed_batch[:-1], transformed_batch[-1])

    def transform_function(self):
        return self._ouput_reuse_transform
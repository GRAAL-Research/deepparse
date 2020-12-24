from typing import List


class TrainVectorizer:

    def __init__(self, embedding_vectorizer, tags_vectorizer):
        """
        Vectorizer use during training to convert an address into word embeddings and to provide the target.
        """
        self.embedding_vectorizer = embedding_vectorizer
        self.tags_vectorizer = tags_vectorizer

    def __call__(self, addresses: List[str]):
        """
        Method to vectorizer addresses for training.

        Args:
            addresses (list[str]): The addresses to vectorize.

        Return:
            A tuple compose of embeddings word addresses' and the target idxs.
        """
        input_sequence = []
        target_sequence = []
        input_sequence.extend(self.embedding_vectorizer([address[0]
                                                         for address in addresses]))  # need to be pass in batch
        # otherwise the padding for byte-pair encoding will be broken
        for address in addresses:
            target_tmp = [self.tags_vectorizer(target) for target in address[1]]
            target_tmp.append(self.tags_vectorizer("EOS"))  # to append the End Of Sequence token
            target_sequence.append(target_tmp)
        return zip(input_sequence, target_sequence)

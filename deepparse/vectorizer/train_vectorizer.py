class TrainVectorizer:

    def __init__(self, embedding_vectorizer, tags_vectorizer):
        self.embedding_vectorizer = embedding_vectorizer
        self.tags_vectorizer = tags_vectorizer

    def __call__(self, addresses):
        input_sequence = []
        target_sequence = []
        for address in addresses:
            input_sequence.extend(self.embedding_vectorizer([address[0]]))
            target_tmp = [self.tags_vectorizer(target) for target in address[1]]
            target_tmp.append(self.tags_vectorizer("EOS"))
            target_sequence.append(target_tmp)
        return list(zip(input_sequence, target_sequence))

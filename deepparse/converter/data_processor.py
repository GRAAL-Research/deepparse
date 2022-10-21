from typing import Callable, List, Tuple

from . import TagsConverter
from ..vectorizer import Vectorizer


class DataProcessor:
    def __init__(
        self,
        vectorizer: Vectorizer,
        sequences_padding_callback: Callable,
        batch_padding_callback: Callable,
        tags_converter: TagsConverter,
    ) -> None:
        self.vectorizer = vectorizer
        self.sequences_padding_callback = sequences_padding_callback
        self.batch_padding_callback = batch_padding_callback
        self.tags_converter = tags_converter

    def process_for_inference(self, addresses: List[str]):
        return self.sequences_padding_callback(self.vectorizer(addresses))

    def process_for_training(self, addresses_and_targets: List[Tuple[str, List[str]]]):
        # TODO: add teacher forcing
        input_sequence = []
        target_sequence = []
        # TODO: use zip to seperate the addresses and targets
        input_sequence.extend(self.vectorizer([address[0] for address in addresses_and_targets]))

        for _, target_list in addresses_and_targets:
            target_tmp = [self.tags_converter(target) for target in target_list]
            target_tmp.append(self.tags_converter("EOS"))  # to append the End Of Sequence token
            target_sequence.append(target_tmp)

        return self.batch_padding_callback(list(zip(input_sequence, target_sequence)))

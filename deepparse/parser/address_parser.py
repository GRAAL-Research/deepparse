import math
import os
import re
from typing import List, Union, Dict

import numpy as np
import torch
from numpy.core.multiarray import ndarray
from poutyne.framework import Experiment
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from .parsed_address import ParsedAddress
from .. import load_tuple_to_device
from ..converter import TagsConverter, fasttext_data_padding, DataTransform
from ..converter.data_padding import bpemb_data_padding
from ..dataset_container import DatasetContainerInterface
from ..embeddings_models import BPEmbEmbeddingsModel
from ..embeddings_models import FastTextEmbeddingsModel
from ..fasttext_tools import download_fasttext_embeddings
from ..metrics import nll_loss_function, accuracy
from ..network.pre_trained_bpemb_seq2seq import PreTrainedBPEmbSeq2SeqModel
from ..network.pre_trained_fasttext_seq2seq import PreTrainedFastTextSeq2SeqModel
from ..vectorizer import FastTextVectorizer, BPEmbVectorizer, TrainVectorizer

_pre_trained_tags_to_idx = {
    "StreetNumber": 0,
    "StreetName": 1,
    "Unit": 2,
    "Municipality": 3,
    "Province": 4,
    "PostalCode": 5,
    "Orientation": 6,
    "GeneralDelivery": 7,
    "EOS": 8  # the 9th is the EOS with idx 8
}


class AddressParser:
    """
    Address parser to parse an address or a list of address using one of the seq2seq pre-trained
    networks either with fastText or BPEmb.

    Args:
        model (str): The network name to use, can be either:

            - fasttext (need ~9 GO of RAM to be used);
            - bpemb (need ~2 GO of RAM to be used);
            - lightest (less RAM usage) (equivalent to bpemb);
            - fastest (quicker to process one address) (equivalent to fasttext);
            - best (best accuracy performance) (equivalent to bpemb).

            The default value is 'best' for the most accurate model.
        device (Union[int, str, torch.device]): The device to use can be either:

            - a ``GPU`` index in int format (e.g. ``0``);
            - a complete device name in a string format (e.g. ``'cuda:0'``);
            - a :class:`~torch.torch.device` object;
            - ``'cpu'`` for a  ``CPU`` use.

            The default value is GPU with the index ``0`` if it exist, otherwise the value is ``CPU``.
        rounding (int): The rounding to use when asking the probability of the tags. The default value is 4 digits.

    Note:
        For both the networks, we will download the pre-trained weights and embeddings in the ``.cache`` directory
        for the root user. Also, one can download all the dependencies of our pre-trained model using the
        `deepparse.download` module (e.g. python -m deepparse.download fasttext) before sending it to a node without
        access to Internet.

    Note:
        Also note that the first time the fastText model is instantiated on a computer, we download the fastText
        pre-trained embeddings of 6.8 GO, and this process can be quite long (a couple of minutes).

    Note:
        The predictions tags are the following

            - "StreetNumber": for the street number
            - "StreetName": for the name of the street
            - "Unit": for the unit (such as apartment)
            - "Municipality": for the municipality
            - "Province": for the province or local region
            - "PostalCode": for the postal code
            - "Orientation": for the street orientation (e.g. west, east)
            - "GeneralDelivery": for other delivery information

    Example:

        .. code-block:: python

                address_parser = AddressParser(device=0) #on gpu device 0
                parse_address = address_parser('350 rue des Lilas Ouest Quebec city Quebec G1L 1B6')

                address_parser = AddressParser(model='fasttext', device='cpu') # fasttext model on cpu
                parse_address = address_parser('350 rue des Lilas Ouest Quebec city Quebec G1L 1B6')
    """

    def __init__(self, model: str = 'best', device: Union[int, str, torch.device] = 0, rounding: int = 4) -> None:
        self._process_device(device)

        self.rounding = rounding

        self.tags_converter = TagsConverter(_pre_trained_tags_to_idx)

        self.model = model.lower()
        # model factory
        if self.model in ("fasttext", "fastest"):
            path = os.path.join(os.path.expanduser('~'), ".cache", "deepparse")
            os.makedirs(path, exist_ok=True)

            file_name = download_fasttext_embeddings("fr", saving_dir=path)
            embeddings_model = FastTextEmbeddingsModel(file_name)

            self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = fasttext_data_padding

            self.pre_trained_model = PreTrainedFastTextSeq2SeqModel(self.device)

        elif self.model in ("bpemb", "best", "lightest"):
            self.vectorizer = BPEmbVectorizer(embeddings_model=BPEmbEmbeddingsModel(lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.pre_trained_model = PreTrainedBPEmbSeq2SeqModel(self.device)
        else:
            raise NotImplementedError(f"There is no {model} network implemented. Value can be: "
                                      f"fasttext, bpemb, lightest (bpemb), fastest (fasttext) or best (bpemb).")

        self.pre_trained_model.eval()

    def __call__(self,
                 addresses_to_parse: Union[List[str], str],
                 with_prob: bool = False) -> Union[ParsedAddress, List[ParsedAddress]]:
        """
        Callable method to parse the components of an address or a list of address.

        Args:
            addresses_to_parse (Union[list[str], str]): The addresses to be parse, can be either a single address
                (when using str) or a list of address. When using a list of addresses, the addresses are processed in
                batch, allowing a faster process. For example, using fastText model, a single address takes around
                0.003 seconds to be parsed using a batch of 1 (1 element at the time is processed).
                This time can be reduced to 0.00035 seconds per address when using a batch of 128
                (128 elements at the time are processed).
            with_prob (bool): If true, return the probability of all the tags with the specified
                rounding.

        Return:
            Either a :class:`~deepparse.deepparse.parsed_address.ParsedAddress` or a list of
            :class:`~deepparse.deepparse.parsed_address.ParsedAddress` when given more than one address.


        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    parse_address = address_parser('350 rue des Lilas Ouest Quebec city Quebec G1L 1B6')
                    parse_address = address_parser('350 rue des Lilas Ouest Quebec city Quebec G1L 1B6', with_prob=True)

        """
        if isinstance(addresses_to_parse, str):
            addresses_to_parse = [addresses_to_parse]

        # since training data is lowercase
        lower_cased_addresses_to_parse = [address.lower() for address in addresses_to_parse]

        vectorize_address = self.vectorizer(lower_cased_addresses_to_parse)

        padded_address = self.data_converter(vectorize_address)
        padded_address = load_tuple_to_device(padded_address, self.device)

        predictions = self.pre_trained_model(*padded_address)

        tags_predictions = predictions.max(2)[1].transpose(0, 1).cpu().numpy()
        tags_predictions_prob = torch.exp(predictions.max(2)[0]).transpose(0, 1).detach().cpu().numpy()

        tagged_addresses_components = self._fill_tagged_addresses_components(tags_predictions, tags_predictions_prob,
                                                                             addresses_to_parse, with_prob)

        return tagged_addresses_components

    def retrain(self,
                dataset_container: DatasetContainerInterface,
                train_ratio: float,
                batch_size: int,
                epochs: int,
                num_workers: int = 1,
                learning_rate: float = 0.1,
                callbacks: Union[List, None] = None,
                seed: int = 42,
                logging_path: str = "./chekpoints") -> List[Dict]:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        """
        Method to retrain our pre-trained model using a dataset with the same tags. We train using
        `experiment <https://poutyne.org/experiment.html>`_ from `poutyne <https://poutyne.org/index.html>`_
        framework. The experiment module allow us to save checkpoint ``ckpt`` (pickle format) and a log.tsv were
        the best epochs can be found (the best epoch is use in test).

        Args:
            dataset_container (~deepparse.deepparse.dataset_container.dataset_container.DatasetContainerInterface): The
                dataset container of the data to use.
            train_ratio (float): The ratio to use of the dataset for the training. The rest of the data is use for valid
                (e.g. a train ratio of 0.8 mean a 80-20 train-valid split).
            batch_size (int): The size of the batch.
            epochs (int): number of training epoch.
            num_workers (int): Number of worker to use for the data loader (default is 1 worker).
            learning_rate (float): The learning rate to use for training. One can also use
                `learning rate callback <https://poutyne.org/callbacks.html#lr-schedulers>`_ to modify the learning
                rate during training.
            callbacks (Union[List, None]): List of callback to use during training.
                See `poutyne <https://poutyne.org/callbacks.html#callback-class>`_ framework for information. By default
                we set no callback.
            seed (int): Seed to use (by default 42).
            logging_path (str): The logging path for the checkpoints. By default the path is ``./chekpoints``.

        Return:
            A list of dictionary with the best epoch stats (see poutyne for example).

        Note:
            We use SGD optimizer, NLL loss and accuracy as in the `article <https://arxiv.org/abs/2006.16152>`_.
            The data are shuffled.
            We use teacher forcing during training (with a prob of 0.5).

        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    data_path = 'path_to_a_pickle_dataset.p'

                    container = PickleDatasetContainer(data_path)

                    address_parser.retrain(container, 0.8, epochs=1, batch_size=128)

        """
        callbacks = [] if callbacks is None else callbacks
        train_generator, valid_generator = self._create_training_data_generator(dataset_container, train_ratio,
                                                                                batch_size, num_workers)

        optimizer = SGD(self.pre_trained_model.parameters(), learning_rate)

        loss_fn = nll_loss_function
        accuracy_fn = accuracy

        exp = Experiment(logging_path,
                         self.pre_trained_model,
                         device=self.device,
                         optimizer=optimizer,
                         loss_function=loss_fn,
                         batch_metrics=[accuracy_fn])

        train_res = exp.train(train_generator,
                              valid_generator=valid_generator,
                              epochs=epochs,
                              seed=seed,
                              callbacks=callbacks)
        return train_res

    def test(self,
             test_dataset_container: DatasetContainerInterface,
             batch_size: int,
             num_workers: int = 1,
             callbacks: Union[List, None] = None,
             seed: int = 42,
             logging_path: str = "./chekpoints",
             checkpoint: Union[str, int] = "best") -> Dict:
        """
        Method to test a retrained or a pre-trained model using a dataset with the same tags. We train using
        `experiment <https://poutyne.org/experiment.html>`_ from `poutyne <https://poutyne.org/index.html>`_
        framework. The experiment module allow us to save checkpoint ``ckpt`` (pickle format) and a log.tsv were
        the best epochs can be found (the best epoch is use in test).

        Args:
            test_dataset_container (~deepparse.deepparse.dataset_container.dataset_container.DatasetContainerInterface): The
                test dataset container of the data to use.
            callbacks (Union[List, None]): List of callback to use during training.
                See `poutyne <https://poutyne.org/callbacks.html#callback-class>`_ framework for information. By default
                we set no callback.
            seed (int): Seed to use (by default 42).
            logging_path (str): The logging path for the checkpoints. By default the path is ``./chekpoints``.
            checkpoint (Union[str, int]): Checkpoint to use for the test. If 'best', will load the best weights.
                If 'last', will load the last model checkpoint. If int, will load the checkpoint of the specified epoch.
                Meaning that the API restrict that your model to load must have a name following format
                ``checkpoint_epoch_<int>.ckpt`` due to framework constraint. (Default value = 'best')
        Return:
            A dictionary with the best epoch stats (see poutyne for example).

        Note:
            We use NLL loss and accuracy as in the `article <https://arxiv.org/abs/2006.16152>`_.

        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    data_path = 'path_to_a_pickle_test_dataset.p'

                    test_container = PickleDatasetContainer(data_path)

                    address_parser.test(test_container) # using the default best epoch
                    address_parser.test(test_container, checkpoint='last') # using the last epoch
                    address_parser.test(test_container, checkpoint=5) # using the epoch 5 model
        """
        callbacks = [] if callbacks is None else callbacks
        data_transform = self._set_data_transformer()

        test_generator = DataLoader(test_dataset_container,
                                    collate_fn=data_transform.output_transform,
                                    batch_size=batch_size,
                                    num_workers=num_workers)
        loss_fn = nll_loss_function
        accuracy_fn = accuracy

        exp = Experiment(logging_path,
                         self.pre_trained_model,
                         device=self.device,
                         loss_function=loss_fn,
                         batch_metrics=[accuracy_fn])

        test_res = exp.test(test_generator, seed=seed, callbacks=callbacks)
        return test_res

    def _fill_tagged_addresses_components(self, tags_predictions: ndarray, tags_predictions_prob: ndarray,
                                          addresses_to_parse: List[str],
                                          with_prob: bool) -> Union[ParsedAddress, List[ParsedAddress]]:
        """
        Method to fill the mapping for every address between a address components and is associated predicted tag (or
        tag and prob).
        """
        tagged_addresses_components = []

        for address_to_parse, tags_prediction, tags_prediction_prob in zip(addresses_to_parse, tags_predictions,
                                                                           tags_predictions_prob):
            tagged_address_components = []
            for word, predicted_idx_tag, tag_proba in zip(address_to_parse.split(), tags_prediction,
                                                          tags_prediction_prob):
                tag = self.tags_converter(predicted_idx_tag)
                if with_prob:
                    tag = (tag, round(tag_proba, self.rounding))
                tagged_address_components.append((word, tag))
            tagged_addresses_components.append(ParsedAddress({address_to_parse: tagged_address_components}))

        if len(tagged_addresses_components) == 1:
            return tagged_addresses_components[0]
        return tagged_addresses_components

    def _process_device(self, device: Union[int, str, torch.device]):
        """
        Function to process the device depending of the argument type.

        Set the device as a torch device object.
        """
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            if re.fullmatch(r'cpu|cuda:\d+', device.lower()):
                self.device = torch.device(device)
            else:
                raise ValueError("String value should be 'cpu' or follow the pattern 'cuda:[int]'.")
        elif isinstance(device, int):
            if device >= 0:
                self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")
            else:
                raise ValueError("Device should not be a negative number.")
        else:
            raise ValueError("Device should be a string, an int or a torch device.")

    def _set_data_transformer(self):
        train_vectorizer = TrainVectorizer(self.vectorizer, self.tags_converter)  # vectorize to provide also the target
        data_transform = DataTransform(train_vectorizer, self.model)  # use for transforming the data prior to training
        return data_transform

    def _create_training_data_generator(self, dataset_container: DatasetContainerInterface, train_ratio: float,
                                        batch_size: int, num_workers: int):
        data_transform = self._set_data_transformer()

        num_data = len(dataset_container)
        indices = list(range(num_data))
        np.random.shuffle(indices)

        split = math.floor(train_ratio * num_data)

        train_indices = indices[:split]
        train_dataset = Subset(dataset_container, train_indices)

        valid_indices = indices[split:]
        valid_dataset = Subset(dataset_container, valid_indices)

        train_generator = DataLoader(train_dataset,
                                     collate_fn=data_transform.teacher_forcing_transform,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=True)

        valid_generator = DataLoader(valid_dataset,
                                     collate_fn=data_transform.output_transform,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

        return train_generator, valid_generator

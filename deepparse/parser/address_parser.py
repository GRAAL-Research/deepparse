import re
from typing import List, Union, Dict, Tuple

import torch
from poutyne.framework import Experiment
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset

from .parsed_address import ParsedAddress
from .. import CACHE_PATH, handle_checkpoint, indices_splitting
from .. import load_tuple_to_device, download_fasttext_magnitude_embeddings
from ..converter import TagsConverter
from ..converter import fasttext_data_padding, bpemb_data_padding, DataTransform
from ..dataset_container import DatasetContainer
from ..embeddings_models import BPEmbEmbeddingsModel
from ..embeddings_models import FastTextEmbeddingsModel
from ..embeddings_models import MagnitudeEmbeddingsModel
from ..fasttext_tools import download_fasttext_embeddings
from ..metrics import nll_loss, accuracy
from ..network.bpemb_seq2seq import BPEmbSeq2SeqModel
from ..network.fasttext_seq2seq import FastTextSeq2SeqModel
from ..vectorizer import FastTextVectorizer, BPEmbVectorizer
from ..vectorizer import TrainVectorizer
from ..vectorizer.magnitude_vectorizer import MagnitudeVectorizer

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

# this threshold represent at which point the prediction of the address takes enough time to
# justify a predict verbosity.
PREDICTION_TIME_PERFORMANCE_THRESHOLD = 64


class AddressParser:
    """
    Address parser to parse an address or a list of address using one of the seq2seq pre-trained
    networks either with fastText or BPEmb.

    Args:
        model_type (str): The network name to use, can be either:

            - fasttext (need ~9 GO of RAM to be used);
            - fasttext-light (need ~2 GO of RAM to be used, but slower than fasttext version);
            - bpemb (need ~2 GO of RAM to be used);
            - fastest (quicker to process one address) (equivalent to fasttext);
            - lightest (the one using the less RAM and GPU usage) (equivalent to fasttext-light);
            - best (best accuracy performance) (equivalent to bpemb).

            The default value is "best" for the most accurate model.
        device (Union[int, str, torch.device]): The device to use can be either:

            - a ``GPU`` index in int format (e.g. ``0``);
            - a complete device name in a string format (e.g. ``'cuda:0'``);
            - a :class:`~torch.torch.device` object;
            - ``'cpu'`` for a  ``CPU`` use.

            The default value is GPU with the index ``0`` if it exist, otherwise the value is ``CPU``.
        rounding (int): The rounding to use when asking the probability of the tags. The default value is 4 digits.
        verbose (bool): Turn on/off the verbosity of the model weights download and loading. The default value is True.
        path_to_retrained_model (Union[str, None]): The path to the retrained model to use for prediction. Be sure to
            use the same model_type as the retrained model. For example, if you fine-tuned our pretrained fasttext model
            and want to use it, model_type='fasttext' and path_to_retrained_model='/path/to/fasttext_model/'.
            Since the architecture of fasttext and fasttext-light are similar, one can interchange both
            (e.g. retrain a fasttext model and load the weights into a fasttext-light model).
            Default is None, meaning we use our pre-trained model.


    Note:
        For both the networks, we will download the pre-trained weights and embeddings in the ``.cache`` directory
        for the root user. The pre-trained weights take at most 44 MB. The fastText embeddings take 6.8 GO,
        the fastText-light embeddings take 3.3 GO and bpemb take 116 MB (in .cache/bpemb).

        Also, one can download all the dependencies of our pre-trained model using the
        `deepparse.download` module (e.g. python -m deepparse.download fasttext) before sending it to a node without
        access to Internet.

        Here are the URLs to download our pre-trained models directly

            - `FastText <https://graal.ift.ulaval.ca/public/deepparse/fasttext.ckpt>`_
            - `BPEmb <https://graal.ift.ulaval.ca/public/deepparse/bpemb.ckpt>`_
            - `FastText Light <https://graal.ift.ulaval.ca/public/deepparse/fasttext.magnitude.gz>`_.

    Note:
        Note that the first time the fastText model is instantiated on a computer, we download the fastText
        pre-trained embeddings of 6.8 GO, and this process can be quite long (a couple of minutes).

    Note:
        You may observe a 100% CPU load the first time you call the fasttext-light model. We
        `hypotheses <https://github.com/GRAAL-Research/deepparse/pull/54#issuecomment-743463855>`_ that this is due
        to the SQLite database behind `pymagnitude`. This approach create a cache to speed up processing and since the
        memory mapping is save between the runs, it's more intensive the first time you call it and subsequent
        time this load doesn't appear.

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
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")

                address_parser = AddressParser(model_type="fasttext", device="cpu") # fasttext model on cpu
                parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")

        Using a retrain model

        .. code-block:: python

                address_parser = AddressParser(model_type="fasttext",
                                               path_to_retrained_model='/path_to_a_retrain_fasttext_model')
    """

    def __init__(self,
                 model_type: str = "best",
                 device: Union[int, str, torch.device] = 0,
                 rounding: int = 4,
                 verbose: bool = True,
                 path_to_retrained_model: Union[str, None] = None) -> None:
        # pylint: disable=too-many-arguments
        self._process_device(device)

        self.rounding = rounding
        self.verbose = verbose

        self.tags_converter = TagsConverter(_pre_trained_tags_to_idx)

        self.model_type = model_type.lower()
        self._model_factory(path_to_retrained_model=path_to_retrained_model)
        self.model.eval()

    def __str__(self) -> str:
        return f"{self.model_type.capitalize()}AddressParser"

    __repr__ = __str__  # to call __str__ when list of address

    def __call__(self,
                 addresses_to_parse: Union[List[str], str],
                 with_prob: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 0) -> Union[ParsedAddress, List[ParsedAddress]]:
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
            batch_size (int): The size of the batch (default is 32).
            num_workers (int): Number of workers to use for the data loader (default is 0, which means that the data
                will be loaded in the main process.).


        Return:
            Either a :class:`~deepparse.deepparse.parsed_address.ParsedAddress` or a list of
            :class:`~deepparse.deepparse.parsed_address.ParsedAddress` when given more than one address.


        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6")
                    parse_address = address_parser("350 rue des Lilas Ouest Quebec city Quebec G1L 1B6",
                                                    with_prob=True)

            Using a larger batch size

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    parse_address = address_parser(a_large_dataset, batch_size=1024)
                    # You can also use more worker
                    parse_address = address_parser(a_large_dataset, batch_size=1024, num_workers=2)

        """
        if isinstance(addresses_to_parse, str):
            addresses_to_parse = [addresses_to_parse]

        # since training data is lowercase
        lower_cased_addresses_to_parse = [address.lower() for address in addresses_to_parse]

        if self.verbose and len(addresses_to_parse) > PREDICTION_TIME_PERFORMANCE_THRESHOLD:
            print("Vectorizing the address")

        predict_data_loader = DataLoader(lower_cased_addresses_to_parse,
                                         collate_fn=self._predict_pipeline,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

        tags_predictions = []
        tags_predictions_prob = []
        for x in predict_data_loader:
            tensor_prediction = self.model(*load_tuple_to_device(x, self.device))
            tags_predictions.extend(tensor_prediction.max(2)[1].transpose(0, 1).cpu().numpy().tolist())
            tags_predictions_prob.extend(
                torch.exp(tensor_prediction.max(2)[0]).transpose(0, 1).detach().cpu().numpy().tolist())

        tagged_addresses_components = self._fill_tagged_addresses_components(tags_predictions, tags_predictions_prob,
                                                                             addresses_to_parse, with_prob)

        return tagged_addresses_components

    def retrain(self,
                dataset_container: DatasetContainer,
                train_ratio: float,
                batch_size: int,
                epochs: int,
                num_workers: int = 1,
                learning_rate: float = 0.01,
                callbacks: Union[List, None] = None,
                seed: int = 42,
                logging_path: str = "./chekpoints") -> List[Dict]:
        # pylint: disable=too-many-arguments, line-too-long, too-many-locals
        """
        Method to retrain the address parser model using a dataset with the same tags. We train using
        `experiment <https://poutyne.org/experiment.html>`_ from `poutyne <https://poutyne.org/index.html>`_
        framework. The experiment module allow us to save checkpoints ``ckpt`` (pickle format) and a log.tsv where
        the best epochs can be found (the best epoch is used in test).

        Args:
            dataset_container (~deepparse.deepparse.dataset_container.dataset_container.DatasetContainer): The
                dataset container of the data to use.
            train_ratio (float): The ratio to use of the dataset for the training. The rest of the data is used for the validation
                (e.g. a train ratio of 0.8 mean a 80-20 train-valid split).
            batch_size (int): The size of the batch.
            epochs (int): number of training epochs.
            num_workers (int): Number of workers to use for the data loader (default is 1 worker).
            learning_rate (float): The learning rate (LR) to use for training (default 0.01). To reduce the LR during
                training, use `Poutyne learning rate scheduler callback
                <https://github.com/GRAAL-Research/poutyne/blob/master/poutyne/framework/callbacks/lr_scheduler.py>`_.
            callbacks (Union[List, None]): List of callbacks to use during training.
                See Poutyne `callback <https://poutyne.org/callbacks.html#callback-class>`_ for more information. By default
                we set no callback.
            seed (int): Seed to use (by default 42).
            logging_path (str): The logging path for the checkpoints. By default the path is ``./chekpoints``.

        Return:
            A list of dictionary with the best epoch stats (see `Experiment class
            <https://poutyne.org/experiment.html#poutyne.Experiment.train>`_ for details).

        Note:
            We use SGD optimizer, NLL loss and accuracy as a metric, the data is shuffled and we use teacher forcing during
            training (with a prob of 0.5) as in the `article <https://arxiv.org/abs/2006.16152>`_.

        Note:
            Due to pymagnitude, we could not train using the Magnitude embeddings, meaning it's not possible to
            train using the fasttext-light model. But, since we don't update the embeddings weights, one can retrain
            using the fasttext model and later on use the weights with the fasttext-light.

        Example:

            .. code-block:: python

                    address_parser = AddressParser(device=0) #on gpu device 0
                    data_path = 'path_to_a_pickle_dataset.p'

                    container = PickleDatasetContainer(data_path)

                    address_parser.retrain(container, 0.8, epochs=1, batch_size=128)

            Using learning rate scheduler callback.

            .. code-block:: python

                    import poutyne

                    address_parser = AddressParser(device=0)
                    data_path = 'path_to_a_pickle_dataset.p'

                    container = PickleDatasetContainer(data_path)

                    lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1) # reduce LR by a factor of 10 each epoch
                    address_parser.retrain(container, 0.8, epochs=5, batch_size=128, callbacks=[lr_scheduler])

        See `this <https://github.com/GRAAL-Research/deepparse/blob/master/examples/fine_tuning.py>`_ for a fine
        tuning example.
        """
        if self.model_type == "fasttext-light":
            raise ValueError("It's not possible to retrain a fasttext-light due to pymagnitude problem.")

        callbacks = [] if callbacks is None else callbacks
        train_generator, valid_generator = self._create_training_data_generator(dataset_container,
                                                                                train_ratio,
                                                                                batch_size,
                                                                                num_workers,
                                                                                seed=seed)

        optimizer = SGD(self.model.parameters(), learning_rate)

        exp = Experiment(logging_path,
                         self.model,
                         device=self.device,
                         optimizer=optimizer,
                         loss_function=nll_loss,
                         batch_metrics=[accuracy])

        train_res = exp.train(train_generator,
                              valid_generator=valid_generator,
                              epochs=epochs,
                              seed=seed,
                              callbacks=callbacks)
        return train_res

    def test(self,
             test_dataset_container: DatasetContainer,
             batch_size: int,
             num_workers: int = 1,
             callbacks: Union[List, None] = None,
             seed: int = 42,
             logging_path: str = "./chekpoints",
             checkpoint: Union[str, int] = "best") -> Dict:
        # pylint: disable=too-many-arguments
        """
        Method to test a retrained or a pre-trained model using a dataset with the same tags. We train using
        `experiment <https://poutyne.org/experiment.html>`_ from `poutyne <https://poutyne.org/index.html>`_
        framework. The experiment module allow us to save checkpoints ``ckpt`` (pickle format) and a log.tsv where
        the best epochs can be found (the best epoch is use in test).

        Args:
            test_dataset_container (~deepparse.deepparse.dataset_container.dataset_container.DatasetContainer):
                The test dataset container of the data to use.
            callbacks (Union[List, None]): List of callbacks to use during training.
                See Poutyne `callback <https://poutyne.org/callbacks.html#callback-class>`_ for more information.
                By default we set no callback.
            seed (int): Seed to use (by default 42).
            logging_path (str): The logging path for the checkpoints. By default the path is ``./chekpoints``.
                checkpoint (Union[str, int]): Checkpoint to use for the test.
                - If 'best', will load the best weights.
                - If 'last', will load the last model checkpoint.
                - If int, will load a specific checkpoint (e.g. 3).
                - If 'str', will load a specific model (e.g. a retrained model), must be a path to a pickled format
                model i.e. ends with a '.p' extension (e.g. retrained_model.p).
                - If 'fasttext', will load our pre-trained fasttext model and test it on your data.
                (Need to have Poutyne>=1.2 to work)
                - If 'bpemb', will load our pre-trained bpemb model and test it on your data.
                (Need to have Poutyne>=1.2 to work)
        Return:
            A dictionary with the best epoch stats (see `Experiment class
            <https://poutyne.org/experiment.html#poutyne.Experiment.train>`_ for details).

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
        if self.model_type == "fasttext-light":
            raise ValueError("It's not possible to test a fasttext-light due to pymagnitude problem.")

        callbacks = [] if callbacks is None else callbacks
        data_transform = self._set_data_transformer()

        test_generator = DataLoader(test_dataset_container,
                                    collate_fn=data_transform.output_transform,
                                    batch_size=batch_size,
                                    num_workers=num_workers)

        exp = Experiment(logging_path, self.model, device=self.device, loss_function=nll_loss, batch_metrics=[accuracy])

        checkpoint = handle_checkpoint(checkpoint)

        test_res = exp.test(test_generator, seed=seed, callbacks=callbacks, checkpoint=checkpoint)
        return test_res

    def _fill_tagged_addresses_components(self, tags_predictions: List, tags_predictions_prob: List,
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
            if re.fullmatch(r"cpu|cuda:\d+", device.lower()):
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
        data_transform = DataTransform(train_vectorizer,
                                       self.model_type)  # use for transforming the data prior to training
        return data_transform

    def _create_training_data_generator(self, dataset_container: DatasetContainer, train_ratio: float, batch_size: int,
                                        num_workers: int, seed: int) -> Tuple:
        # pylint: disable=too-many-arguments
        data_transform = self._set_data_transformer()

        train_indices, valid_indices = indices_splitting(num_data=len(dataset_container),
                                                         train_ratio=train_ratio,
                                                         seed=seed)

        train_dataset = Subset(dataset_container, train_indices)
        train_generator = DataLoader(train_dataset,
                                     collate_fn=data_transform.teacher_forcing_transform,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=True)

        valid_dataset = Subset(dataset_container, valid_indices)
        valid_generator = DataLoader(valid_dataset,
                                     collate_fn=data_transform.output_transform,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

        return train_generator, valid_generator

    def _model_factory(self, path_to_retrained_model: Union[str, None] = None) -> None:
        """
        Model factory to create the vectorizer, the data converter and the pre-trained model
        """
        if self.model_type in ("fasttext", "fastest", "fasttext-light", "lightest"):
            if self.model_type in ("fasttext-light", "lightest"):
                self.model_type = "fasttext-light"  # we change name to 'fasttext-light' since name can be lightest
                file_name = download_fasttext_magnitude_embeddings(saving_dir=CACHE_PATH, verbose=self.verbose)

                embeddings_model = MagnitudeEmbeddingsModel(file_name, verbose=self.verbose)
                self.vectorizer = MagnitudeVectorizer(embeddings_model=embeddings_model)
            else:
                self.model_type = "fasttext"  # we change name to fasttext since name can be fastest
                file_name = download_fasttext_embeddings(saving_dir=CACHE_PATH, verbose=self.verbose)

                embeddings_model = FastTextEmbeddingsModel(file_name, verbose=self.verbose)
                self.vectorizer = FastTextVectorizer(embeddings_model=embeddings_model)

            self.data_converter = fasttext_data_padding

            self.model = FastTextSeq2SeqModel(self.device,
                                              verbose=self.verbose,
                                              path_to_retrained_model=path_to_retrained_model)

        elif self.model_type in ("bpemb", "best"):
            self.model_type = "bpemb"  # we change name to bpemb since name can be best
            self.vectorizer = BPEmbVectorizer(
                embeddings_model=BPEmbEmbeddingsModel(verbose=self.verbose, lang="multi", vs=100000, dim=300))

            self.data_converter = bpemb_data_padding

            self.model = BPEmbSeq2SeqModel(self.device,
                                           verbose=self.verbose,
                                           path_to_retrained_model=path_to_retrained_model)
        else:
            raise NotImplementedError(f"There is no {self.model_type} network implemented. Value should be: "
                                      f"fasttext, bpemb, lightest (fasttext-light), fastest (fasttext) "
                                      f"or best (bpemb).")

    def _predict_pipeline(self, data):
        """
        Pipeline to process data in a data loader for prediction.
        """
        return self.data_converter(self.vectorizer(data))

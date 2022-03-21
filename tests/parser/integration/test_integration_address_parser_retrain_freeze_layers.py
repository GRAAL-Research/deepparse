# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
from unittest import skipIf

from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserIntegrationTestFreezeLayers(AddressParserRetrainTestCase):
    def assert_layer_frozen(self, model_part):
        # A frozen layer does not requires grad
        for param in model_part.parameters():
            self.assertFalse(param.requires_grad)

    def assert_layer_not_frozen(self, model_part):
        # A frozen layer does requires grad
        for param in model_part.parameters():
            self.assertTrue(param.requires_grad)

    def test_givenEncoderToFreeze_thenFreezeLayer(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="encoder",
        )

        self.assert_layer_frozen(address_parser.model.encoder.lstm)
        self.assert_layer_not_frozen(address_parser.model.decoder.lstm)
        self.assert_layer_not_frozen(address_parser.model.decoder.linear)

    def test_givenDecoderToFreeze_thenFreezeLayer(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="decoder",
        )

        self.assert_layer_not_frozen(address_parser.model.encoder.lstm)
        self.assert_layer_frozen(address_parser.model.decoder.lstm)
        self.assert_layer_not_frozen(address_parser.model.decoder.linear)

    def test_givenDecoderBPEmbToFreeze_thenFreezeEmbeddingsLayer(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="decoder",
        )

        self.assert_layer_frozen(address_parser.model.embedding_network)

    def test_givenSeq2SeqToFreeze_thenFreezeLayer(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="seq2seq",
        )

        self.assert_layer_frozen(address_parser.model.encoder.lstm)
        self.assert_layer_frozen(address_parser.model.decoder.lstm)
        self.assert_layer_not_frozen(address_parser.model.decoder.linear)

    def test_givenSeq2SeqBPEmbToFreeze_thenFreezeEmbeddingsLayer(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="seq2seq",
        )

        self.assert_layer_frozen(address_parser.model.embedding_network)

    def test_givenLinearToFreeze_thenFreezeLayer(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )
        address_parser.retrain(
            self.training_container,
            self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
            layers_to_freeze="prediction_layer",
        )

        self.assert_layer_not_frozen(address_parser.model.encoder.lstm)
        self.assert_layer_not_frozen(address_parser.model.decoder.lstm)
        self.assert_layer_frozen(address_parser.model.decoder.linear)

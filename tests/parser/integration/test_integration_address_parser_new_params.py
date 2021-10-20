# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from unittest import skipIf

import torch

from deepparse.parser import FormattedParsedAddress, AddressParser
from deepparse.parser import formatted_parsed_address
from tests.parser.integration.base_predict import AddressParserPredictNewParamsBase


@skipIf(not torch.cuda.is_available(), "no gpu available")
# We skip it even if it is CPU since the downloading is too long
class AddressParserPredictTest(AddressParserPredictNewParamsBase):

    def test_givenAAddress_whenParseNewParamsFastTextCPU_thenParseAddressProperly(self):
        # Training setup
        fasttext_address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_cpu_device,
                                                verbose=self.verbose)
        self.training(fasttext_address_parser,
                      self.training_container,
                      self.a_number_of_workers,
                      seq2seq_params=self.seq2seq_params)

        # Test
        fasttext_address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_cpu_device,
                                                verbose=self.verbose,
                                                path_to_retrained_model=self.a_fasttext_retrain_model_path)

        # Since we train a smaller model, it sometime return EOS, so we manage it by adding the EOS tag
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery", "EOS"
        ]
        # We validate that the new settings are loaded and we can parse
        parse_address = fasttext_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAAddress_whenParseNewParamsFastTextGPU_thenParseAddressProperly(self):
        # Training setup
        fasttext_address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_torch_device,
                                                verbose=self.verbose)
        self.training(fasttext_address_parser,
                      self.training_container,
                      self.a_number_of_workers,
                      seq2seq_params=self.seq2seq_params)

        # Test
        fasttext_address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_cpu_device,
                                                verbose=self.verbose,
                                                path_to_retrained_model=self.a_fasttext_retrain_model_path)

        # Since we train a smaller model, it sometime return EOS, so we manage it by adding the EOS tag
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery", "EOS"
        ]
        # We validate that the new settings are loaded and we can parse
        parse_address = fasttext_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAAddress_whenParseNewParamsBPEmbCPU_thenParseAddressProperly(self):
        # Training setup
        bpemb_address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                             device=self.a_cpu_device,
                                             verbose=self.verbose)
        self.training(bpemb_address_parser,
                      self.training_container,
                      self.a_number_of_workers,
                      seq2seq_params=self.seq2seq_params)

        # Test
        bpemb_address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                             device=self.a_cpu_device,
                                             verbose=self.verbose,
                                             path_to_retrained_model=self.a_bpemb_retrain_model_path)

        # Since we train a smaller model, it sometime return EOS, so we manage it by adding the EOS tag
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery", "EOS"
        ]

        # We validate that the new settings are loaded
        parse_address = bpemb_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

    def test_givenAAddress_whenParseNewParamsBPEmbGPU_thenParseAddressProperly(self):
        # Training setup
        bpemb_address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                             device=self.a_torch_device,
                                             verbose=self.verbose)
        self.training(bpemb_address_parser,
                      self.training_container,
                      self.a_number_of_workers,
                      seq2seq_params=self.seq2seq_params)

        # Test
        bpemb_address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                             device=self.a_torch_device,
                                             verbose=self.verbose,
                                             path_to_retrained_model=self.a_bpemb_retrain_model_path)

        # Since we train a smaller model, it sometime return EOS, so we manage it by adding the EOS tag
        formatted_parsed_address.FIELDS = [
            "StreetNumber", "Unit", "StreetName", "Orientation", "Municipality", "Province", "PostalCode",
            "GeneralDelivery", "EOS"
        ]

        # We validate that the new settings are loaded
        parse_address = bpemb_address_parser(self.an_address_to_parse)
        self.assertIsInstance(parse_address, FormattedParsedAddress)

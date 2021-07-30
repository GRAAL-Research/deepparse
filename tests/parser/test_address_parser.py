# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=unused-argument, no-member, too-many-public-methods

import os
import unittest
from unittest import skipIf
from unittest.mock import patch, Mock

import torch
from torch import device

from deepparse.parser import FormattedParsedAddress
from deepparse.parser.address_parser import AddressParser
from tests.parser.base import AddressParserPredictTestCase


class AddressParserTest(AddressParserPredictTestCase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserTest, cls).setUpClass()
        cls.a_BPemb_name = "BpembAddressParser"
        cls.a_fasttext_name = "FasttextAddressParser"
        cls.a_fasttext_light_name = "Fasttext-lightAddressParser"
        cls.a_rounding = 5
        cls.a_cpu_device = "cpu"
        cls.a_cpu_torch_device = device(cls.a_cpu_device)
        cls.a_gpu_device = 0
        cls.a_gpu_torch_device = device(cls.a_gpu_device)
        cls.verbose = False
        cls.number_tags = 9

        cls.correct_address_components = {"ATag": 0, "AnotherTag": 1, "EOS": 2}
        cls.incorrect_address_components = {"ATag": 0, "AnotherTag": 1}

        cls.BPEmb_embeddings_model_param = {"lang": "multi", "vs": 100000, "dim": 300}
        cls.fasttext_download_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")
        os.makedirs(cls.fasttext_download_path, exist_ok=True)
        cls.a_embeddings_path = "."

    def setUp(self):
        super().setUp()
        self.BPEmb_mock = Mock()
        self.fasttext_mock = Mock()

        self.embeddings_model_mock = Mock()

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenACPUDeviceSetup_whenInstantiatingParser_thenDeviceIsCPU(self, embeddings_model_mock, model_mock):
        address_parser = AddressParser(
            model_type=self.a_best_model_type.capitalize(),
            # we use BPEmb for simplicity
            device=self.a_cpu_device)
        actual = address_parser.device
        expected = self.a_cpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(torch.cuda.is_available(), "gpu available so we skip the test")
    # We skip the test since this one test if cuda is not available, we set to CPU by default
    def test_givenAGPUDeviceSetup_whenInstantiatingParserWithoutGPU_thenRaiseWarningAndCPU(
            self, embeddings_model_mock, model_mock):
        with self.assertWarns(UserWarning):
            address_parser = AddressParser(
                model_type=self.a_best_model_type.capitalize(),
                # we use BPEmb for simplicity
                device=self.a_gpu_device)
        actual = address_parser.device
        expected = self.a_cpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenAGPUDeviceSetup_whenInstantiatingParser_thenDeviceIsGPU(self, embeddings_model_mock, model_mock):
        address_parser = AddressParser(
            model_type=self.a_best_model_type.capitalize(),
            # we use BPEmb for simplicity
            device=self.a_gpu_device)
        actual = address_parser.device
        expected = self.a_gpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenAGPUDeviceSetupSTRFormat_whenInstantiatingParser_thenDeviceIsGPU(self, embeddings_model_mock,
                                                                                   model_mock):
        address_parser = AddressParser(
            model_type=self.a_best_model_type.capitalize(),
            # we use BPEmb for simplicity
            device="cuda:0")
        actual = address_parser.device
        expected = self.a_gpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenAGPUDeviceSetupINTFormat_whenInstantiatingParser_thenDeviceIsGPU(self, embeddings_model_mock,
                                                                                   model_mock):
        address_parser = AddressParser(
            model_type=self.a_best_model_type.capitalize(),
            # we use BPEmb for simplicity
            device=0)
        actual = address_parser.device
        expected = self.a_gpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_givenAGPUTorchDeviceSetup_whenInstantiatingParser_thenDeviceIsGPU(self, embeddings_model_mock, model_mock):
        address_parser = AddressParser(
            model_type=self.a_best_model_type.capitalize(),
            # we use BPEmb for simplicity
            device=self.a_gpu_torch_device)
        actual = address_parser.device
        expected = self.a_gpu_torch_device
        self.assertEqual(actual, expected)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenACapitalizeBPEmbModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model_mock:
            AddressParser(model_type=self.a_best_model_type.capitalize(),
                          device=self.a_cpu_device,
                          verbose=self.verbose)

            embeddings_model_mock.assert_called_with(verbose=self.verbose)

        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model_mock:
            AddressParser(model_type=self.a_bpemb_model_type.capitalize(),
                          device=self.a_cpu_device,
                          verbose=self.verbose)

            embeddings_model_mock.assert_called_with(verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenACapitalizeFastTextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader_mock:
            AddressParser(model_type=self.a_fastest_model_type.capitalize(),
                          device=self.a_cpu_device,
                          verbose=self.verbose)

            downloader_mock.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader_mock:
            AddressParser(model_type=self.a_fasttext_model_type.capitalize(),
                          device=self.a_cpu_device,
                          verbose=self.verbose)

            downloader_mock.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model_mock:
            AddressParser(model_type=self.a_best_model_type, device=self.a_cpu_device, verbose=self.verbose)

            embeddings_model_mock.assert_called_with(verbose=self.verbose)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model_mock:
            AddressParser(model_type=self.a_bpemb_model_type, device=self.a_cpu_device, verbose=self.verbose)

            embeddings_model_mock.assert_called_with(verbose=self.verbose)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_best_model_type, device=self.a_cpu_device, verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_bpemb_model_type, device=self.a_cpu_device, verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            AddressParser(model_type=self.a_best_model_type, device=self.a_cpu_device, verbose=self.verbose)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          self.number_tags,
                                          verbose=self.verbose,
                                          path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModelType_whenInstantiatingParserWithUserComponent_thenCorrectNumberOfOutputDim(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.setup_retrain_new_tags_model(self.correct_address_components, self.a_bpemb_model_type)
            AddressParser(model_type=self.a_bpemb_model_type,
                          device=self.a_cpu_device,
                          verbose=self.verbose,
                          path_to_retrained_model=self.a_model_path)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          len(self.correct_address_components),
                                          verbose=self.verbose,
                                          path_to_retrained_model=self.a_model_path)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFasttextModelType_whenInstantiatingParserWithUserComponent_thenCorrectNumberOfOutputDim(
            self, download_weights_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.setup_retrain_new_tags_model(self.incorrect_address_components, self.a_fasttext_model_type)
            AddressParser(model_type=self.a_fasttext_model_type,
                          device=self.a_cpu_device,
                          verbose=self.verbose,
                          path_to_retrained_model=self.a_model_path)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          len(self.incorrect_address_components),
                                          verbose=self.verbose,
                                          path_to_retrained_model=self.a_model_path)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            AddressParser(model_type=self.a_bpemb_model_type, device=self.a_cpu_device, verbose=self.verbose)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          self.number_tags,
                                          verbose=self.verbose,
                                          path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            AddressParser(model_type=self.a_fastest_model_type, device=self.a_cpu_device, verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            AddressParser(model_type=self.a_fasttext_model_type, device=self.a_cpu_device, verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenDownloadFasttextMagnitudeModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings") as downloader:
            AddressParser(model_type=self.a_fasttext_light_model_type, device=self.a_cpu_device, verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model_mock:
                AddressParser(model_type=self.a_fastest_model_type, device=self.a_cpu_device, verbose=self.verbose)

                embeddings_model_mock.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model_mock:
                AddressParser(model_type=self.a_fasttext_model_type, device=self.a_cpu_device, verbose=self.verbose)

                embeddings_model_mock.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings",
                   return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel") as embeddings_model_mock:
                AddressParser(model_type=self.a_fasttext_light_model_type,
                              device=self.a_cpu_device,
                              verbose=self.verbose)

                embeddings_model_mock.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_fastest_model_type, device=self.a_cpu_device, verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_fasttext_model_type, device=self.a_cpu_device, verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_fasttext_light_model_type,
                              device=self.a_cpu_device,
                              verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenALightestModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer_mock:
                AddressParser(model_type=self.a_fasttext_lightest_model_type,
                              device=self.a_cpu_device,
                              verbose=self.verbose)

                vectorizer_mock.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, download_weights_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            AddressParser(model_type=self.a_fastest_model_type, device=self.a_cpu_device, verbose=self.verbose)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          self.number_tags,
                                          verbose=self.verbose,
                                          path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, download_weights_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            AddressParser(model_type=self.a_fasttext_model_type, device=self.a_cpu_device, verbose=self.verbose)

            model_mock.assert_called_with(self.a_cpu_torch_device,
                                          self.number_tags,
                                          verbose=self.verbose,
                                          path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAString_thenParseAddress(self, download_weights_mock,
                                                                            embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, FormattedParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_multiple_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], FormattedParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.GeneralDelivery)
            self.assertEqual(parse_address.Municipality, self.a_municipality)
            self.assertIsNone(parse_address.Orientation)
            self.assertEqual(parse_address.PostalCode, self.a_postal_code)
            self.assertEqual(parse_address.Province, self.a_province)
            self.assertEqual(parse_address.StreetName, self.a_street_name)
            self.assertEqual(parse_address.StreetNumber, self.a_street_number)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAString_thenParseAddress(self, download_weights_mock,
                                                                             embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, FormattedParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_multiple_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], FormattedParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.GeneralDelivery)
            self.assertEqual(parse_address.Municipality, self.a_municipality)
            self.assertIsNone(parse_address.Orientation)
            self.assertEqual(parse_address.PostalCode, self.a_postal_code)
            self.assertEqual(parse_address.Province, self.a_province)
            self.assertEqual(parse_address.StreetName, self.a_street_name)
            self.assertEqual(parse_address.StreetNumber, self.a_street_number)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAString_thenParseAddress(self, embeddings_model_mock,
                                                                         vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, FormattedParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAListOfAddress_thenParseAllAddress(self, embeddings_model_mock,
                                                                                   vectorizer_model_mock,
                                                                                   data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_multiple_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], FormattedParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(self, embeddings_model_mock,
                                                                                    vectorizer_model_mock,
                                                                                    data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)

            parse_address = address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.GeneralDelivery)
            self.assertEqual(parse_address.Municipality, self.a_municipality)
            self.assertIsNone(parse_address.Orientation)
            self.assertEqual(parse_address.PostalCode, self.a_postal_code)
            self.assertEqual(parse_address.Province, self.a_province)
            self.assertEqual(parse_address.StreetName, self.a_street_name)
            self.assertEqual(parse_address.StreetNumber, self.a_street_number)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAnAddressVerbose_thenVerbose(self, embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock):
        self._capture_output()
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            with patch("deepparse.parser.address_parser.PREDICTION_TIME_PERFORMANCE_THRESHOLD", 0):
                self.mock_predictions_vectors(model_mock)
                address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                               device=self.a_cpu_device,
                                               verbose=True)

                address_parser(self.a_complete_address)
                actual = self.test_out.getvalue().strip()
                expect = "Vectorizing the address"
                self.assertEqual(actual, expect)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAnBPembAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser)

            self.assertEqual(self.a_BPemb_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAnBPembAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_best_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser.__repr__())

            self.assertEqual(self.a_BPemb_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser)

            self.assertEqual(self.a_fasttext_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser.__repr__())

            self.assertEqual(self.a_fasttext_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextLightAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_lightest_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser)

            self.assertEqual(self.a_fasttext_light_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextLightAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_fasttext_lightest_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            print(address_parser.__repr__())

            self.assertEqual(self.a_fasttext_light_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModelType_whenRetrainWithIncorrectPredictionTags_thenRaiseValueError(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel"):
            address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            with self.assertRaises(ValueError):
                address_parser.retrain(Mock(), 0.8, 1, 1, prediction_tags=self.incorrect_address_components)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFasttextModelType_whenInstantiatingParserWithUserComponent_thenRaiseValueError(
            self, download_weights_mock, model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel"):
            address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                           device=self.a_cpu_device,
                                           verbose=self.verbose)
            with self.assertRaises(ValueError):
                address_parser.retrain(Mock(), 0.8, 1, 1, prediction_tags=self.incorrect_address_components)

    # we do BPemb but can be fasttext or fasttext-light
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAModel_whenAddressParsingAnAddressVerbose_thenVerbose(self, embeddings_model_mock,
                                                                        vectorizer_model_mock, data_padding_mock):
        self._capture_output()
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            with patch("deepparse.parser.address_parser.PREDICTION_TIME_PERFORMANCE_THRESHOLD", 0):
                self.mock_predictions_vectors(model_mock)
                address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                               device=self.a_cpu_device,
                                               verbose=True)

                address_parser(self.a_complete_address)
                actual = self.test_out.getvalue().strip()
                expect = "Vectorizing the address"
                self.assertEqual(actual, expect)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAModel_whenAddressParsingAnAddressWithProb_thenIncludeProb(self, embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model_mock:
            self.mock_predictions_vectors(model_mock)
            address_parser = AddressParser(model_type=self.a_bpemb_model_type, device=self.a_cpu_device, verbose=True)

            output = address_parser(self.a_complete_address, with_prob=True)
            self.assertIsInstance(output.address_parsed_components[0][1], tuple)  # tuple of prob
            self.assertIsInstance(output.address_parsed_components[1][1], tuple)


if __name__ == "__main__":
    unittest.main()

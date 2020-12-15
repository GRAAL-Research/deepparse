# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=W0613
# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

import os
from unittest import TestCase
from unittest.mock import patch, Mock

import torch
from torch import device, tensor

from deepparse.parser import ParsedAddress
from deepparse.parser.address_parser import AddressParser


class AddressParserTest(TestCase):
    # pylint: disable=too-many-public-methods
    @classmethod
    def setUpClass(cls):
        cls.a_address = "3 test road quebec"
        cls.a_best_model_type = "best"
        cls.a_BPEmb_model_type = "BPEmb"
        cls.a_fastest_model_type = "fastest"
        cls.a_fasttext_model_type = "fasttext"
        cls.a_fasttext_light_model_type = "fasttext-light"
        cls.a_fasttext_lightest_model_type = "lightest"
        cls.a_rounding = 5
        cls.a_device = "cpu"
        cls.a_torch_device = device(cls.a_device)
        cls.verbose = True

        cls.BPEmb_embeddings_model_param = {"lang": "multi", "vs": 100000, "dim": 300}
        cls.fasttext_download_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")
        os.makedirs(cls.fasttext_download_path, exist_ok=True)
        cls.a_embeddings_path = "."

        # here a example with the model prediction vectors
        cls.a_complete_address = "15 major st london ontario n5z1e1"
        cls.a_municipality = "london"
        cls.a_postal_code = "n5z1e1"
        cls.a_province = "ontario"
        cls.a_street_name = "major st"
        cls.a_street_number = "15"
        cls.a_prediction_vector_for_a_complete_address = tensor([[[
            -6.7080e-04, -7.3572e+00, -1.4086e+01, -1.1092e+01, -2.1749e+01, -1.1060e+01, -1.4627e+01, -1.4654e+01,
            -2.8624e+01
        ]],
                                                                 [[
                                                                     -1.5119e+01, -1.7881e-06, -1.7613e+01, -1.3365e+01,
                                                                     -2.9415e+01, -2.3198e+01, -2.2065e+01, -2.2009e+01,
                                                                     -4.0588e+01
                                                                 ]],
                                                                 [[
                                                                     -1.5922e+01, -1.1903e-03, -1.3102e+01, -6.7359e+00,
                                                                     -2.4669e+01, -1.7328e+01, -1.9970e+01, -1.9923e+01,
                                                                     -4.0041e+01
                                                                 ]],
                                                                 [[
                                                                     -1.9461e+01, -1.3808e+01, -1.5707e+01, -2.0146e-05,
                                                                     -1.0881e+01, -1.5345e+01, -2.1945e+01, -2.2081e+01,
                                                                     -4.6854e+01
                                                                 ]],
                                                                 [[
                                                                     -1.7136e+01, -1.8420e+01, -1.5489e+01, -1.5802e+01,
                                                                     -1.2159e-05, -1.1350e+01, -2.1703e+01, -2.1866e+01,
                                                                     -4.2224e+01
                                                                 ]],
                                                                 [[
                                                                     -1.4736e+01, -1.7999e+01, -1.5483e+01, -2.1751e+01,
                                                                     -1.3005e+01, -3.4571e-06, -1.7897e+01, -1.7965e+01,
                                                                     -1.4235e+01
                                                                 ]],
                                                                 [[
                                                                     -1.7509e+01, -1.8191e+01, -1.7853e+01, -2.6309e+01,
                                                                     -1.7179e+01, -1.0518e+01, -1.9438e+01, -1.9542e+01,
                                                                     -2.7060e-05
                                                                 ]]])

    def setUp(self):
        self.address_parser = 0
        self.BPEmb_mock = Mock()
        self.fasttext_mock = Mock()

        self.embeddings_model_mock = Mock()

    def mock_predictions_vectors(self, model):
        model.return_value = Mock(return_value=self.a_prediction_vector_for_a_complete_address)

    def mock_multiple_predictions_vectors(self, model):
        model.return_value = Mock(return_value=torch.cat((self.a_prediction_vector_for_a_complete_address,
                                                          self.a_prediction_vector_for_a_complete_address), 1))

    @patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

    @patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model=self.a_BPEmb_model_type, device=self.a_device)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

    @patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_BPEmb_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiatePretrainedModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel") as model:
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiatePretrainedModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel") as model:
            self.address_parser = AddressParser(model=self.a_BPEmb_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenDownloadFasttextMagnitudeModelWithCorrectPath(
            self, embeddings_model_mock, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings") as downloader:
            self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateModelWithCorrectPath(
            self, pretrained_model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings",
                   return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenALightestModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer:
                self.address_parser = AddressParser(model=self.a_fasttext_lightest_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiatePretrainedModelWithCorrectParameters(
            self, pretrained_model_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiatePretrainedModelWithCorrectParameters(
            self, pretrained_model_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAFasttextModel_whenAddressParsingAString_thenParseAddress(self, pretrained_model_mock,
                                                                            embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAFasttextModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, pretrained_model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAFasttextModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, pretrained_model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.general_delivery)
            self.assertEqual(parse_address.municipality, self.a_municipality)
            self.assertIsNone(parse_address.orientation)
            self.assertEqual(parse_address.postal_code, self.a_postal_code)
            self.assertEqual(parse_address.province, self.a_province)
            self.assertEqual(parse_address.street_name, self.a_street_name)
            self.assertEqual(parse_address.street_number, self.a_street_number)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAString_thenParseAddress(self, pretrained_model_mock,
                                                                             embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, pretrained_model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, pretrained_model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_fasttext_light_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.general_delivery)
            self.assertEqual(parse_address.municipality, self.a_municipality)
            self.assertIsNone(parse_address.orientation)
            self.assertEqual(parse_address.postal_code, self.a_postal_code)
            self.assertEqual(parse_address.province, self.a_province)
            self.assertEqual(parse_address.street_name, self.a_street_name)
            self.assertEqual(parse_address.street_number, self.a_street_number)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAString_thenParseAddress(self, pretrained_model_mock,
                                                                         embeddings_model_mock, vectorizer_model_mock,
                                                                         bpemb_data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAListOfAddress_thenParseAllAddress(self, pretrained_model_mock,
                                                                                   embeddings_model_mock,
                                                                                   vectorizer_model_mock,
                                                                                   bpemb_data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(self, pretrained_model_mock,
                                                                                    embeddings_model_mock,
                                                                                    vectorizer_model_mock,
                                                                                    bpemb_data_padding_mock):
        with patch("deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.general_delivery)
            self.assertEqual(parse_address.municipality, self.a_municipality)
            self.assertIsNone(parse_address.orientation)
            self.assertEqual(parse_address.postal_code, self.a_postal_code)
            self.assertEqual(parse_address.province, self.a_province)
            self.assertEqual(parse_address.street_name, self.a_street_name)
            self.assertEqual(parse_address.street_number, self.a_street_number)

# Since we use a patch as model mock we skip the unused argument error
# pylint: disable=W0613
import os
import unittest
from unittest.mock import patch, Mock

from torch import device

from deepparse.parser import ParsedAddress
from deepparse.parser.address_parser import AddressParser
from tests.parser.base import AddressParserPredictTestCase


class AddressParserTest(AddressParserPredictTestCase):
    # pylint: disable=too-many-public-methods
    @classmethod
    def setUpClass(cls):
        super(AddressParserTest, cls).setUpClass()
        cls.a_BPemb_name = "BpembAddressParser"
        cls.a_fasttext_name = "FasttextAddressParser"
        cls.a_fasttext_light_name = "Fasttext-lightAddressParser"
        cls.a_rounding = 5
        cls.a_device = "cpu"
        cls.a_torch_device = device(cls.a_device)
        cls.verbose = False

        cls.BPEmb_embeddings_model_param = {"lang": "multi", "vs": 100000, "dim": 300}
        cls.fasttext_download_path = os.path.join(os.path.expanduser("~"), ".cache", "deepparse")
        os.makedirs(cls.fasttext_download_path, exist_ok=True)
        cls.a_embeddings_path = "."

    def setUp(self):
        super().setUp()
        self.address_parser = 0
        self.BPEmb_mock = Mock()
        self.fasttext_mock = Mock()

        self.embeddings_model_mock = Mock()

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenACapitalizeBPEmbModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model_type=self.a_best_model_type.capitalize(),
                                                device=self.a_device,
                                                verbose=self.verbose)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type.capitalize(),
                                                device=self.a_device,
                                                verbose=self.verbose)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenACapitalizeFastTextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model_type=self.a_fastest_model_type.capitalize(),
                                                device=self.a_device,
                                                verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type.capitalize(),
                                                device=self.a_device,
                                                verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model_type=self.a_best_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbEmbeddingsModelWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel") as embeddings_model:
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            embeddings_model.assert_called_with(verbose=self.verbose, **self.BPEmb_embeddings_model_param)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_best_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateBPEmbVectorizerWithCorrectParameters(
            self, model_mock):
        with patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.BPEmbVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.address_parser = AddressParser(model_type=self.a_best_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose, path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    def test_givenABPEmbModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose, path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model_type=self.a_fastest_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings") as downloader:
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenDownloadFasttextMagnitudeModelWithCorrectPath(
            self, embeddings_model_mock, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings") as downloader:
            self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            downloader.assert_called_with(saving_dir=self.fasttext_download_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model_type=self.a_fastest_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_embeddings", return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateModelWithCorrectPath(self, model_mock):
        with patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings",
                   return_value=self.a_embeddings_path):
            with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel") as embeddings_model:
                self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                embeddings_model.assert_called_with(self.a_embeddings_path, verbose=self.verbose)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_fastest_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateFasttextVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.FastTextEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.FastTextVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenAFasttextLightModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.FastTextSeq2SeqModel")
    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    # pylint: disable=C0301
    def test_givenALightestModelType_whenInstanciatingParser_thenInstanciateMagnitudeVectorizerWithCorrectParameters(
            self, model_mock, downloader_mock):
        with patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel", return_value=self.embeddings_model_mock):
            with patch("deepparse.parser.address_parser.MagnitudeVectorizer") as vectorizer:
                self.address_parser = AddressParser(model_type=self.a_fasttext_lightest_model_type,
                                                    device=self.a_device,
                                                    verbose=self.verbose)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFastestModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, download_weights_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.address_parser = AddressParser(model_type=self.a_fastest_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose, path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    def test_givenAFasttextModelType_whenInstantiatingParser_thenInstantiateModelWithCorrectParameters(
            self, download_weights_mock, embeddings_model_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            model.assert_called_with(self.a_torch_device, verbose=self.verbose, path_to_retrained_model=None)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAString_thenParseAddress(self, download_weights_mock,
                                                                            embeddings_model_mock,
                                                                            vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAFasttextModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

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
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAString_thenParseAddress(self, download_weights_mock,
                                                                             embeddings_model_mock,
                                                                             vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAListOfAddress_thenParseAllAddress(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAMagnitudeModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.general_delivery)
            self.assertEqual(parse_address.municipality, self.a_municipality)
            self.assertIsNone(parse_address.orientation)
            self.assertEqual(parse_address.postal_code, self.a_postal_code)
            self.assertEqual(parse_address.province, self.a_province)
            self.assertEqual(parse_address.street_name, self.a_street_name)
            self.assertEqual(parse_address.street_number, self.a_street_number)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAString_thenParseAddress(self, embeddings_model_mock,
                                                                         vectorizer_model_mock, data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsInstance(parse_address, ParsedAddress)
            self.assertEqual(parse_address.raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAListOfAddress_thenParseAllAddress(self, embeddings_model_mock,
                                                                                   vectorizer_model_mock,
                                                                                   data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.mock_multiple_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser([self.a_complete_address, self.a_complete_address])

            self.assertIsInstance(parse_address, list)
            self.assertIsInstance(parse_address[0], ParsedAddress)
            self.assertEqual(parse_address[0].raw_address, self.a_complete_address)
            self.assertEqual(parse_address[1].raw_address, self.a_complete_address)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenABPEmbModel_whenAddressParsingAnAddress_thenParseAddressCorrectly(self, embeddings_model_mock,
                                                                                    vectorizer_model_mock,
                                                                                    data_padding_mock):
        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)

            parse_address = self.address_parser(self.a_complete_address)

            self.assertIsNone(parse_address.general_delivery)
            self.assertEqual(parse_address.municipality, self.a_municipality)
            self.assertIsNone(parse_address.orientation)
            self.assertEqual(parse_address.postal_code, self.a_postal_code)
            self.assertEqual(parse_address.province, self.a_province)
            self.assertEqual(parse_address.street_name, self.a_street_name)
            self.assertEqual(parse_address.street_number, self.a_street_number)

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAnBPembAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser)

            self.assertEqual(self.a_BPemb_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.BPEmbEmbeddingsModel")
    @patch("deepparse.parser.address_parser.BPEmbVectorizer")
    @patch("deepparse.parser.address_parser.bpemb_data_padding")
    def test_givenAnBPembAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.BPEmbSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_best_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser.__repr__())

            self.assertEqual(self.a_BPemb_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser)

            self.assertEqual(self.a_fasttext_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_embeddings")
    @patch("deepparse.parser.address_parser.FastTextEmbeddingsModel")
    @patch("deepparse.parser.address_parser.FastTextVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser.__repr__())

            self.assertEqual(self.a_fasttext_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextLightAddressParser_whenStrAddressParser_thenStringIsModelTypeAddressParse(
            self, download_weights_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_lightest_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser)

            self.assertEqual(self.a_fasttext_light_name, self.test_out.getvalue().strip())

    @patch("deepparse.parser.address_parser.download_fasttext_magnitude_embeddings")
    @patch("deepparse.parser.address_parser.MagnitudeEmbeddingsModel")
    @patch("deepparse.parser.address_parser.MagnitudeVectorizer")
    @patch("deepparse.parser.address_parser.fasttext_data_padding")
    def test_givenAnFasttextLightAddressParser_whenReprAddressParser_thenStringIsModelTypeAddressParse(
            self, model_mock, embeddings_model_mock, vectorizer_model_mock, data_padding_mock):
        self._capture_output()

        with patch("deepparse.parser.address_parser.FastTextSeq2SeqModel") as model:
            self.mock_predictions_vectors(model)
            self.address_parser = AddressParser(model_type=self.a_fasttext_lightest_model_type,
                                                device=self.a_device,
                                                verbose=self.verbose)
            print(self.address_parser.__repr__())

            self.assertEqual(self.a_fasttext_light_name, self.test_out.getvalue().strip())


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import TestCase
from unittest.mock import patch, Mock
import os

from torch import device

from deepparse.parser.address_parser import AddressParser


class AddressParserTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_address = '3 test road quebec'
        cls.a_best_model_type = 'best'
        cls.a_bpemb_model_type = 'bpemb'
        cls.a_fastest_model_type = 'fastest'
        cls.a_fasttext_model_type = 'fasttext'
        cls.a_rounding = 5
        cls.a_device = 'cpu'
        cls.a_torch_device = device(cls.a_device)

        cls.bpemb_embeddings_model_param = {'lang': "multi", 'vs': 100000, 'dim': 300}
        cls.fasttext_download_path = os.path.join(os.path.expanduser('~'), ".cache", "deepparse")
        os.makedirs(cls.fasttext_download_path, exist_ok=True)
        cls.a_language = 'fr'
        cls.a_embeddings_path = '.'

    def setUp(self):
        self.address_parser = 0
        self.bpemb_mock = Mock()
        self.fasttext_mock = Mock()

        self.embeddings_model_mock = Mock()

    @patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel')
    def test_givenABestModelType_whenInstanciatingParser_thenShouldInstanciateBpembEmbeddingsModelWithCorrectParameters(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel') as embeddings_model:
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            embeddings_model.assert_called_with(**self.bpemb_embeddings_model_param)

    @patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel')
    def test_givenABpembModelType_whenInstanciatingParser_thenShouldInstanciateBpembEmbeddingsModelWithCorrectParameters(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel') as embeddings_model:
            self.address_parser = AddressParser(model=self.a_bpemb_model_type, device=self.a_device)

            embeddings_model.assert_called_with(**self.bpemb_embeddings_model_param)

    @patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel')
    def test_givenABestModelType_whenInstanciatingParser_thenShouldInstanciateBpembVectorizerWithCorrectParameters(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel', return_value=self.embeddings_model_mock):
            with patch('deepparse.parser.address_parser.BPEmbVectorizer') as vectorizer:
                self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel')
    def test_givenABpembModelType_whenInstanciatingParser_thenShouldInstanciateBpembVectorizerWithCorrectParameters(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel', return_value=self.embeddings_model_mock):
            with patch('deepparse.parser.address_parser.BPEmbVectorizer') as vectorizer:
                self.address_parser = AddressParser(model=self.a_bpemb_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel')
    def test_givenABestModelType_whenInstanciatingParser_thenShouldInstanciatePretrainedModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel') as model:
            self.address_parser = AddressParser(model=self.a_best_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device)

    @patch('deepparse.parser.address_parser.BPEmbEmbeddingsModel')
    def test_givenABpembModelType_whenInstanciatingParser_thenShouldInstanciatePretrainedModelWithCorrectParameters(
            self, embeddings_model_mock):
        with patch('deepparse.parser.address_parser.PreTrainedBPEmbSeq2SeqModel') as model:
            self.address_parser = AddressParser(model=self.a_bpemb_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device)

    @patch('deepparse.parser.address_parser.FastTextEmbeddingsModel')
    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    def test_givenAFastestModelType_whenInstanciatingParser_thenShouldDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.download_fasttext_embeddings') as downloader:
            self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

            downloader.assert_called_with(self.a_language, saving_dir=self.fasttext_download_path)

    @patch('deepparse.parser.address_parser.FastTextEmbeddingsModel')
    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    def test_givenAFasttextModelType_whenInstanciatingParser_thenShouldDownloadFasttextModelWithCorrectPath(
            self, embeddings_model_mock, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.download_fasttext_embeddings') as downloader:
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            downloader.assert_called_with(self.a_language, saving_dir=self.fasttext_download_path)

    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    def test_givenAFastestModelType_whenInstanciatingParser_thenShouldInstanciateModelWithCorrectPath(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.download_fasttext_embeddings', return_value=self.a_embeddings_path):
            with patch('deepparse.parser.address_parser.FastTextEmbeddingsModel') as embeddings_model:
                self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

                embeddings_model.assert_called_with(self.a_embeddings_path)

    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    def test_givenAFasttextModelType_whenInstanciatingParser_thenShouldInstanciateModelWithCorrectPath(
            self, pretrained_model_mock):
        with patch('deepparse.parser.address_parser.download_fasttext_embeddings', return_value=self.a_embeddings_path):
            with patch('deepparse.parser.address_parser.FastTextEmbeddingsModel') as embeddings_model:
                self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

                embeddings_model.assert_called_with(self.a_embeddings_path)

    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    @patch('deepparse.parser.address_parser.download_fasttext_embeddings')
    def test_givenAFastestModelType__whenInstanciatingParser_thenShouldInstanciateFasttextVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch('deepparse.parser.address_parser.FastTextEmbeddingsModel', return_value=self.embeddings_model_mock):
            with patch('deepparse.parser.address_parser.FastTextVectorizer') as vectorizer:
                self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel')
    @patch('deepparse.parser.address_parser.download_fasttext_embeddings')
    def test_givenAFasttextModelType__whenInstanciatingParser_thenShouldInstanciateFasttextVectorizerWithCorrectParameters(
            self, pretrained_model_mock, downloader_mock):
        with patch('deepparse.parser.address_parser.FastTextEmbeddingsModel', return_value=self.embeddings_model_mock):
            with patch('deepparse.parser.address_parser.FastTextVectorizer') as vectorizer:
                self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

                vectorizer.assert_called_with(embeddings_model=self.embeddings_model_mock)

    @patch('deepparse.parser.address_parser.download_fasttext_embeddings')
    @patch('deepparse.parser.address_parser.FastTextEmbeddingsModel')
    def test_givenAFastestModelType__whenInstanciatingParser_thenShouldInstanciatePretrainedModelrWithCorrectParameters(
            self, pretrained_model_mock, embeddings_model_mock):
        with patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel') as model:
            self.address_parser = AddressParser(model=self.a_fastest_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device)

    @patch('deepparse.parser.address_parser.download_fasttext_embeddings')
    @patch('deepparse.parser.address_parser.FastTextEmbeddingsModel')
    def test_givenAFasttextModelType__whenInstanciatingParser_thenShouldInstanciatePretrainedModelrWithCorrectParameters(
            self, pretrained_model_mock, embeddings_model_mock):
        with patch('deepparse.parser.address_parser.PreTrainedFastTextSeq2SeqModel') as model:
            self.address_parser = AddressParser(model=self.a_fasttext_model_type, device=self.a_device)

            model.assert_called_with(self.a_torch_device)

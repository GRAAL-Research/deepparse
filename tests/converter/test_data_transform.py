import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch, call

from deepparse.converter import DataTransform, fasttext_data_padding_teacher_forcing, \
    fasttext_data_padding_with_target, bpemb_data_padding_teacher_forcing, bpemb_data_padding_with_target


class DataTransformTest(TestCase):

    def setUp(self) -> None:
        self.train_vectorizer_mock = MagicMock()
        self.a_fastest_model_type = "fastest"
        self.a_fasttext_model_type = "fasttext"
        self.a_bpemb_model_type = "bpemb"
        self.a_best_model_type = "best"
        self.a_fasttext_light_model_type = "fasttext-light"

    def test_whenInstantiateAFastTextDataTransform_thenParametersAreOk(self):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_fasttext_model_type)

        # teacher forcing padding test
        expected = fasttext_data_padding_teacher_forcing
        self.assertIs(expected, data_transform.teacher_forcing_data_padding_fn)

        # output transform padding test
        expected = fasttext_data_padding_with_target
        self.assertIs(expected, data_transform.output_transform_data_padding_fn)

    def test_whenInstantiateAFastestDataTransform_thenParametersAreOk(self):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_fastest_model_type)

        # teacher forcing padding test
        expected = fasttext_data_padding_teacher_forcing
        self.assertIs(expected, data_transform.teacher_forcing_data_padding_fn)

        # output transform padding test
        expected = fasttext_data_padding_with_target
        self.assertIs(expected, data_transform.output_transform_data_padding_fn)

    def test_whenInstantiateABPEmbDataTransform_thenParametersAreOk(self):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_bpemb_model_type)

        # teacher forcing padding test
        expected = bpemb_data_padding_teacher_forcing
        self.assertIs(expected, data_transform.teacher_forcing_data_padding_fn)

        # output transform padding test
        expected = bpemb_data_padding_with_target
        self.assertIs(expected, data_transform.output_transform_data_padding_fn)

    def test_whenInstantiateABestDataTransform_thenParametersAreOk(self):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_best_model_type)

        # teacher forcing padding test
        expected = bpemb_data_padding_teacher_forcing
        self.assertIs(expected, data_transform.teacher_forcing_data_padding_fn)

        # output transform padding test
        expected = bpemb_data_padding_with_target
        self.assertIs(expected, data_transform.output_transform_data_padding_fn)

    def test_whenInstantiateAFasttextLightDataTransform_thenRaiseError(self):
        with self.assertRaises(NotImplementedError):
            _ = DataTransform(self.train_vectorizer_mock, self.a_fasttext_light_model_type)

    @patch("deepparse.converter.data_transform.fasttext_data_padding_teacher_forcing")
    def test_givenAFasttextDataTransform_whenTeacherForcingTransform_thenComponentsAreCall(self, teacher_forcing_mock):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_fasttext_model_type)

        batch_pairs_mock = MagicMock()

        data_transform.teacher_forcing_transform(batch_pairs_mock)

        train_vectorizer_call = [call(batch_pairs_mock)]
        self.train_vectorizer_mock.assert_has_calls(train_vectorizer_call)

        train_vectorizer_call = [call(self.train_vectorizer_mock())]
        teacher_forcing_mock.assert_has_calls(train_vectorizer_call)

    @patch("deepparse.converter.data_transform.bpemb_data_padding_teacher_forcing")
    def test_givenABPEmbDataTransform_whenTeacherForcingTransform_thenComponentsAreCall(self, teacher_forcing_mock):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_bpemb_model_type)

        batch_pairs_mock = MagicMock()

        data_transform.teacher_forcing_transform(batch_pairs_mock)

        train_vectorizer_call = [call(batch_pairs_mock)]
        self.train_vectorizer_mock.assert_has_calls(train_vectorizer_call)

        train_vectorizer_call = [call(self.train_vectorizer_mock())]
        teacher_forcing_mock.assert_has_calls(train_vectorizer_call)

    @patch("deepparse.converter.data_transform.fasttext_data_padding_with_target")
    def test_givenAFasttextDataTransform_whenOutputTransform_thenComponentsAreCall(self, output_transform_mock):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_fasttext_model_type)

        batch_pairs_mock = MagicMock()

        data_transform.output_transform(batch_pairs_mock)

        train_vectorizer_call = [call(batch_pairs_mock)]
        self.train_vectorizer_mock.assert_has_calls(train_vectorizer_call)

        train_vectorizer_call = [call(self.train_vectorizer_mock())]
        output_transform_mock.assert_has_calls(train_vectorizer_call)

    @patch("deepparse.converter.data_transform.bpemb_data_padding_with_target")
    def test_givenABPEmbDataTransform_whenOutputTransform_thenComponentsAreCall(self, output_transform_mock):
        data_transform = DataTransform(self.train_vectorizer_mock, self.a_bpemb_model_type)

        batch_pairs_mock = MagicMock()

        data_transform.output_transform(batch_pairs_mock)

        train_vectorizer_call = [call(batch_pairs_mock)]
        self.train_vectorizer_mock.assert_has_calls(train_vectorizer_call)

        train_vectorizer_call = [call(self.train_vectorizer_mock())]
        output_transform_mock.assert_has_calls(train_vectorizer_call)


if __name__ == "__main__":
    unittest.main()

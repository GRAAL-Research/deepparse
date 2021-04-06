# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# We also skip protected-access since we test the encoder and decoder step
# pylint: disable=not-callable, too-many-public-methods

import os
import pickle
import unittest
from unittest import skipIf
from unittest.mock import MagicMock, call, ANY

import torch
from poutyne import Callback

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser
from tests.parser.integration.base_integration import AddressParserRetrainTestCase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserIntegrationNewPredictionLayerTest(AddressParserRetrainTestCase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserIntegrationNewPredictionLayerTest, cls).setUpClass()
        cls.address_components = {"ATag": 0, "AnotherTag": 1, "EOS": 2}
        cls.prediction_tags_file_name = "prediction_tags.p"
        cls.expected_output_dim = len(cls.address_components)

        cls.class_training_setup()

    @classmethod
    def class_training_setup(cls):
        file_extension = "p"
        training_dataset_name = "sample_incomplete_data_new_prediction_tags"
        test_dataset_name = "test_sample_data_new_prediction_tags"
        download_from_url(training_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)
        download_from_url(test_dataset_name, cls.a_data_saving_dir, file_extension=file_extension)

        cls.training_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, training_dataset_name + "." + file_extension))
        cls.test_container = PickleDatasetContainer(
            os.path.join(cls.a_data_saving_dir, test_dataset_name + "." + file_extension))

    def tearDown(self) -> None:
        super().tearDown()

        self.prediction_tags_dict_tear_down()

    def prediction_tags_dict_setup(self):
        os.makedirs(self.a_checkpoints_saving_dir, exist_ok=True)
        with open(os.path.join(self.a_checkpoints_saving_dir, self.prediction_tags_file_name), "wb") as file:
            pickle.dump(self.address_components, file)

    def prediction_tags_dict_tear_down(self):
        if os.path.exists(os.path.join(self.a_checkpoints_saving_dir, self.prediction_tags_file_name)):
            os.remove(os.path.join(self.a_checkpoints_saving_dir, self.prediction_tags_file_name))

        if os.path.exists(self.prediction_tags_file_name):
            os.remove(self.prediction_tags_file_name)

    # Retrain API tests
    def test_givenAFasttextAddressParser_whenRetrainWithNewPredictionTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithNewPredictionTags_thenPredictionLayerDimChange(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        _ = address_parser.retrain(self.training_container,
                                   self.a_train_ratio,
                                   epochs=self.a_single_epoch,
                                   batch_size=self.a_batch_size,
                                   num_workers=self.a_number_of_workers,
                                   logging_path=self.a_checkpoints_saving_dir,
                                   prediction_tags=self.address_components)
        self.assertEqual(self.expected_output_dim, address_parser.model.output_size)

    def test_givenAFasttextAddressParser_whenRetrainMultipleEpochsAndNewPredictionTags_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfigAndNewPredictionTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithConfigWithCallbacksAndNewPredictionTags_thenCallbackAreUse(
            self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextLightAddressParser_whenRetrainWithNewPredictionTags_thenTrainingDoesNotOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_light_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        with self.assertRaises(ValueError):
            _ = address_parser.retrain(self.training_container,
                                       self.a_train_ratio,
                                       epochs=self.a_single_epoch,
                                       batch_size=self.a_batch_size,
                                       num_workers=self.a_number_of_workers,
                                       logging_path=self.a_checkpoints_saving_dir,
                                       prediction_tags=self.address_components)

    def test_givenABPEmbAddressParser_whenRetrainWithNewPredictionTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithNewPredictionTags_thenPredictionLayerDimChange(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        _ = address_parser.retrain(self.training_container,
                                   self.a_train_ratio,
                                   epochs=self.a_single_epoch,
                                   batch_size=self.a_batch_size,
                                   num_workers=self.a_number_of_workers,
                                   logging_path=self.a_checkpoints_saving_dir,
                                   prediction_tags=self.address_components)

        self.assertEqual(self.expected_output_dim, address_parser.model.output_size)

    def test_givenABPEmbAddressParser_whenRetrainMultipleEpochsAndNewPredictionTags_thenTrainingOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_three_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfigAndNewPredictionTags_thenTrainingOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

    def test_givenABPEmbAddressParser_whenRetrainWithConfigWithCallbacksAndNewPredictionTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        callback_mock = MagicMock(spec=Callback)
        performance_after_training = address_parser.retrain(self.training_container,
                                                            self.a_train_ratio,
                                                            epochs=self.a_single_epoch,
                                                            batch_size=self.a_batch_size,
                                                            num_workers=self.a_number_of_workers,
                                                            learning_rate=self.a_learning_rate,
                                                            callbacks=[callback_mock],
                                                            logging_path=self.a_checkpoints_saving_dir,
                                                            prediction_tags=self.address_components)

        self.assertIsNotNone(performance_after_training)

        callback_train_start_call = [call.on_train_begin({})]
        callback_mock.assert_has_calls(callback_train_start_call)
        callback_train_end_call = [call.on_train_end({})]
        callback_mock.assert_has_calls(callback_train_end_call)
        callback_mock.assert_not_called()

    # Test API tests
    def test_givenAFasttextAddressParser_whenTestWithNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithNewPredictionTags_thenLoadProperNewTagsDict(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, with_new_prediction_tags=self.address_components)

        _ = address_parser.test(self.test_container,
                                batch_size=self.a_batch_size,
                                num_workers=self.a_number_of_workers,
                                logging_path=self.a_checkpoints_saving_dir)

        self.assertEqual(self.address_components, address_parser.tags_converter.tags_to_idx)

    def test_givenAFasttextAddressParser_whenTestMultipleEpochsWithNewPredictionTags_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithConfigWithCallbacksAndNewPredictionTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock],
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end({
                "time": ANY,
                "test_loss": performance_after_test["test_loss"],
                "test_accuracy": performance_after_test["test_accuracy"]
            })
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenAFasttextAddressParser_whenTestWithIntCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.a_single_epoch)

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithLastCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="last")

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithFasttextCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.prediction_tags_dict_setup()

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="fasttext")

        self.assertIsNotNone(performance_after_test)

    def test_givenAFasttextAddressParser_whenTestWithStrCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_fasttext_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, with_new_prediction_tags=self.address_components)

        str_path = os.path.join(self.a_checkpoints_saving_dir, "checkpoint_epoch_1.ckpt")

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=str_path)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithNewPredictionTags_thenLoadProperNewTagsDict(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)
        self.training(address_parser, with_new_prediction_tags=self.address_components)

        _ = address_parser.test(self.test_container,
                                batch_size=self.a_batch_size,
                                num_workers=self.a_number_of_workers,
                                logging_path=self.a_checkpoints_saving_dir)

        self.assertEqual(self.address_components, address_parser.tags_converter.tags_to_idx)

    def test_givenABPEmbAddressParser_whenTestMultipleEpochsWithNewPredictionTags_thenTestOccurCorrectly(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithConfigWithCallbacksAndNewPredictionTags_thenCallbackAreUse(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        callback_mock = MagicMock()
        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     callbacks=[callback_mock],
                                                     logging_path=self.a_checkpoints_saving_dir)

        self.assertIsNotNone(performance_after_test)

        callback_test_start_call = [call.on_test_begin({})]
        callback_mock.assert_has_calls(callback_test_start_call)
        callback_test_end_call = [
            call.on_test_end({
                "time": ANY,
                "test_loss": performance_after_test["test_loss"],
                "test_accuracy": performance_after_test["test_accuracy"]
            })
        ]
        callback_mock.assert_has_calls(callback_test_end_call)
        callback_mock.assert_not_called()

    def test_givenABPEmbAddressParser_whenTestWithIntCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=self.a_single_epoch)

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithLastCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="last")

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithBPEmbCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.prediction_tags_dict_setup()

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint="bpemb")

        self.assertIsNotNone(performance_after_test)

    def test_givenABPEmbAddressParser_whenTestWithStrCkptAndNewPredictionTags_thenTestOccur(self):
        address_parser = AddressParser(model_type=self.a_bpemb_model_type,
                                       device=self.a_torch_device,
                                       verbose=self.verbose)

        self.training(address_parser, with_new_prediction_tags=self.address_components)

        str_path = os.path.join(self.a_checkpoints_saving_dir, "checkpoint_epoch_1.ckpt")

        performance_after_test = address_parser.test(self.test_container,
                                                     batch_size=self.a_batch_size,
                                                     num_workers=self.a_number_of_workers,
                                                     logging_path=self.a_checkpoints_saving_dir,
                                                     checkpoint=str_path)

        self.assertIsNotNone(performance_after_test)


if __name__ == "__main__":
    unittest.main()

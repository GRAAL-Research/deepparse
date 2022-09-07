import os
import unittest
from unittest import skipIf

from deepparse.parser import AddressParser
from tests.base_capture_output import CaptureOutputTestCase
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserIntegrationRetrainAPITest(AddressParserRetrainTestCase, CaptureOutputTestCase):
    def test_givenAFasttextAddressParser_whenRetrainNoValDataset_thenTrainingOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_training = address_parser.retrain(
            self.training_container,
            val_dataset_container=None,
            train_ratio=self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
        )

        self.assertIsNotNone(performance_after_training)

    def test_givenAFasttextAddressParser_whenRetrainWithValDataset_thenTrainingOccur(self):
        address_parser = AddressParser(
            model_type=self.a_fasttext_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        performance_after_training = address_parser.retrain(
            self.training_container,
            val_dataset_container=self.training_container,
            train_ratio=self.a_train_ratio,
            epochs=self.a_single_epoch,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
            logging_path=self.a_checkpoints_saving_dir,
        )

        self.assertIsNotNone(performance_after_training)


if __name__ == "__main__":
    unittest.main()

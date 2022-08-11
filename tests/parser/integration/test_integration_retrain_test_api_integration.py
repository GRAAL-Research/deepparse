import os
from unittest import skipIf

from deepparse.parser import AddressParser
from tests.parser.integration.base_retrain import AddressParserRetrainTestCase


@skipIf(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "deepparse", "cc.fr.300.bin")),
    "download of model too long for test in runner",
)
class AddressParserIntegrationTestAPITest(AddressParserRetrainTestCase):
    def test_givenARetrainAnTestLoop_whenRunBoth_thenWork(self):
        address_parser = AddressParser(
            model_type=self.a_bpemb_model_type,
            device=self.a_cpu_device,
            verbose=self.verbose,
        )

        self.training(address_parser, self.training_container, num_workers=self.a_number_of_workers)

        performance_after_test = address_parser.test(
            self.test_container,
            batch_size=self.a_batch_size,
            num_workers=self.a_number_of_workers,
        )

        self.assertIsNotNone(performance_after_test)

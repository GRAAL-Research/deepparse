from unittest import TestCase

from deepparse.pre_processing import (
    coma_cleaning,
    lower_cleaning,
    trailing_whitespace_cleaning,
    double_whitespaces_cleaning,
    PreProcessorList,
)


class PreProcessorListTest(TestCase):
    def setUp(self):
        self.a_clean_address = ["350 rue des lilas ouest québec québec g1l 1b6"]
        self.a_dirty_address_with_commas = ["350 rue des lilas , Ouest, Québec Québec, G1L 1B6"]
        # We use the default address cleaner
        self.pre_processors = [coma_cleaning, lower_cleaning, trailing_whitespace_cleaning, double_whitespaces_cleaning]

    def test_givenALowerCaseWithComa_whenApply_thenReturnLowerCaseNoComa(self):
        pre_processor_list = PreProcessorList(self.pre_processors)

        actual = pre_processor_list.apply(self.a_dirty_address_with_commas)

        expected = self.a_clean_address

        self.assertListEqual(actual, expected)

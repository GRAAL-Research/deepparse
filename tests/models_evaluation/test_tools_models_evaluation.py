import unittest
from unittest import TestCase

from models_evaluation.tools import clean_up_name, train_country_file, zero_shot_eval_country_file


class ToolsTests(TestCase):

    def setUp(self) -> None:
        self.a_list_of_country_name_to_reformat = ["Korea", "Venezuela republic country", "Russian Federation"]
        self.reformatted_country_names = ["South Korea", "Venezuela", "Russia"]
        self.a_list_of_country_name_not_to_reformat = ["Canada", "Ireland", "Mexico", "Australia"]
        self.some_train_test_files = ["br.p", "us.p", "kp.p", "ru.p", "de.p", "fr.p"]
        self.some_zero_shot_test_files = ["ie.p", "rs.p", "uz.p", "ua.p", "za.p", "py.p", "gr.p"]

    def test_givenSomeNonFormattedCountryName_whenCleanUpName_thenReformatThem(self):
        for country_to_reformat, reformatted_country_name in zip(self.a_list_of_country_name_to_reformat,
                                                                 self.reformatted_country_names):
            actual = clean_up_name(country_to_reformat)
            expected = reformatted_country_name

            self.assertEqual(expected, actual)

    def test_givenSomeCountryName_whenCleanUpName_thenReturnSameName(self):
        for country_to_reformat, reformatted_country_name in zip(self.a_list_of_country_name_not_to_reformat,
                                                                 self.a_list_of_country_name_not_to_reformat):
            actual = clean_up_name(country_to_reformat)
            expected = reformatted_country_name

            self.assertEqual(expected, actual)

    def test_givenATrainCountryFile_whenIsTrainCountryFile_thenReturnTrue(self):
        for train_test_file in self.some_train_test_files:
            self.assertTrue(train_country_file(train_test_file))

    def test_givenANonTrainCountryFile_whenIsTrainCountryFile_thenReturnFalse(self):
        for zero_shot_test_file in self.some_zero_shot_test_files:
            self.assertFalse(train_country_file(zero_shot_test_file))

    def test_givenAZeroShotCountryFile_whenIsZeroShotCountryFile_thenReturnTrue(self):
        for zero_shot_test_file in self.some_zero_shot_test_files:
            self.assertTrue(zero_shot_eval_country_file(zero_shot_test_file))

    def test_givenANonZeroShotCountryFile_whenIsZeroShotCountryFile_thenReturnFalse(self):
        for zero_shot_test_file in self.some_zero_shot_test_files:
            self.assertFalse(train_country_file(zero_shot_test_file))


if __name__ == "__main__":
    unittest.main()

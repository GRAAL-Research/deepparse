from unittest import TestCase

from deepparse.dataset_container import former_python_list


class DatasetContainerToolsTests(TestCase):
    def test_given_a_str_list_split_comma_and_whitespace_when_parse_return_properly_parse_list(self):
        a_list = [0, 1, 2, 3]
        str_list = str(a_list)
        expected_parsing = [str(el) for el in a_list]

        actual_parsing = former_python_list(str_list)
        self.assertEqual(expected_parsing, actual_parsing)

    def test_given_a_str_list_split_comma_when_parse_return_properly_parse_list(self):
        a_list = [0, 1, 2, 3]
        str_list = "[0,1,2,3]"
        expected_parsing = [str(el) for el in a_list]

        actual_parsing = former_python_list(str_list)
        self.assertEqual(expected_parsing, actual_parsing)

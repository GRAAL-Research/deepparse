import unittest
from unittest import TestCase

from deepparse.converter import TagsConverter


class TargetConverterTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_TAG = 'first_tag'
        cls.A_INDEX = 1
        cls.ANOTHER_TAG = 'second_tag'
        cls.ANOTHER_INDEX = 2

        cls.TAG_TO_IDX = {cls.A_TAG: cls.A_INDEX, cls.ANOTHER_TAG: cls.ANOTHER_INDEX}

    def setUp(self):
        self.target_converter = TagsConverter(self.TAG_TO_IDX)

    def test_whenCalledWithString_thenShouldReturnIndex(self):
        index = self.target_converter(self.A_TAG)

        self.assertEqual(index, self.A_INDEX)

    def test_whenCalledWithInt_thenShouldReturnTag(self):
        tag = self.target_converter(self.ANOTHER_INDEX)

        self.assertEqual(tag, self.ANOTHER_TAG)


if __name__ == '__main__':
    unittest.main()
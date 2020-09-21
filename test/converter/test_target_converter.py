import unittest
from unittest import TestCase

from deepparse.converter.target_converter import TagsConverter


class TargetConverterTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first_tag = 'first_tag'
        cls.first_index = 1
        cls.second_tag = 'second_tag'
        cls.second_index = 2

        cls.tags_to_idx = {cls.first_tag: cls.first_index, cls.second_tag: cls.second_index}

    def setUp(self):
        self.target_converter = TagsConverter(self.tags_to_idx)

    def test_whenCalledWithString_thenShouldReturnIndex(self):
        index = self.target_converter(self.first_tag)

        self.assertEquals(index, self.first_index)

    def test_whenCalledWithInt_thenShouldReturnTag(self):
        tag = self.target_converter(self.second_index)
        
        self.assertEquals(tag, self.second_tag)

if __name__ == '__main__':
    unittest.main()
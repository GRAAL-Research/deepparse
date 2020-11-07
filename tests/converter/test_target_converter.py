import unittest
from unittest import TestCase

from deepparse.converter import TagsConverter


class TargetConverterTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a_tag = "first_tag"
        cls.a_index = 1
        cls.another_tag = "second_tag"
        cls.another_index = 2

        cls.tag_to_idx = {cls.a_tag: cls.a_index, cls.another_tag: cls.another_index}

    def setUp(self):
        self.target_converter = TagsConverter(self.tag_to_idx)

    def test_whenCalledWithString_thenShouldReturnIndex(self):
        index = self.target_converter(self.a_tag)

        self.assertEqual(index, self.a_index)

    def test_whenCalledWithInt_thenShouldReturnTag(self):
        tag = self.target_converter(self.another_index)

        self.assertEqual(tag, self.another_tag)


if __name__ == "__main__":
    unittest.main()

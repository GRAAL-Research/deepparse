# Bug with PyTorch source code makes torch.tensor as not callable for pylint.

from unittest import TestCase

from deepparse.parser import AddressParser


class AddressParserPredictBase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fasttext_address_parser = AddressParser(model_type="fasttext", device='cpu')
        cls.bpemb_address_parser = AddressParser(model_type="bpemb", device='cpu')

    def setUp(self):
        self.an_address_to_parse = "350 rue des lilas o"

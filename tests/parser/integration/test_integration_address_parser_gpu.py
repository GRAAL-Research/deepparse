# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from unittest import skipIf

import torch

from tests.parser.integration.base_predict import AddressParserPredictBase


@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserPredictGPUTest(AddressParserPredictBase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserPredictGPUTest, cls).setUpClass()
        cls.device = torch.device("cuda:0")

    def test_givenAAddress_whenParseFastText_thenParseAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse)
        self.assert_properly_parse(parse_address)

    def test_givenAAddress_whenParseBPEmb_thenParseAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse)
        self.assert_properly_parse(parse_address)

    def test_givenAAddress_whenParseFastTextAtt_thenParseAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False, "attention_mechanism": True}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse)
        self.assert_properly_parse(parse_address)

    def test_givenAAddress_whenParseBPEmbAtt_thenParseAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False, "attention_mechanism": True}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse)
        self.assert_properly_parse(parse_address)

    def test_givenAListOfAddress_whenParseFastText_thenParseAllAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse])
        self.assert_properly_parse(parse_address, multiple_address=True)

    def test_givenAListOfAddress_whenParseBPEmb_thenParseAllAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse])
        self.assert_properly_parse(parse_address, multiple_address=True)


# test if num_workers > 0 is correct for the data loader
@skipIf(not torch.cuda.is_available(), "no gpu available")
class AddressParserPredictGPUMultiProcessTest(AddressParserPredictBase):

    @classmethod
    def setUpClass(cls):
        super(AddressParserPredictGPUMultiProcessTest, cls).setUpClass()
        cls.device = torch.device("cuda:0")

    def test_givenAAddress_whenParseFastTextNumWorkers1_thenParseAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=1)
        self.assert_properly_parse(parse_address)

    def test_givenAAddress_whenParseBPEmbNumWorkers1_thenParseAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=1)
        self.assert_properly_parse(parse_address)

    def test_givenAListOfAddress_whenParseFastTextNumWorkers1_thenParseAllAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse], num_workers=1)
        self.assert_properly_parse(parse_address, multiple_address=True)

    def test_givenAListOfAddress_whenParseBPEmbNumWorkers1_thenParseAllAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse], num_workers=1)
        self.assert_properly_parse(parse_address, multiple_address=True)

    def test_givenAAddress_whenParseFastTextNumWorkers2_thenParseAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=2)
        self.assert_properly_parse(parse_address)

    def test_givenAAddress_whenParseBPEmbNumWorkers2_thenParseAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=2)
        self.assert_properly_parse(parse_address)

    def test_givenAListOfAddress_whenParseFastTextNumWorkers2_thenParseAllAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse], num_workers=2)
        self.assert_properly_parse(parse_address, multiple_address=True)

    def test_givenAListOfAddress_whenParseBPEmbNumWorkers2_thenParseAllAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False}
        self.setup_model_with_config(config)

        parse_address = self.a_model([self.an_address_to_parse, self.an_address_to_parse], num_workers=2)
        self.assert_properly_parse(parse_address, multiple_address=True)

    def test_givenAAttentionModel_whenParseFastTextNumWorkers2_thenProperlyParseAddress(self):
        config = {"model_type": "fasttext", "device": self.device, "verbose": False, "attention_mechanism": True}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=2)
        self.assert_properly_parse(parse_address)

    def test_givenAAttentionModel_whenParseBPEmbNumWorkers2_thenProperlyParseAddress(self):
        config = {"model_type": "bpemb", "device": self.device, "verbose": False, "attention_mechanism": True}
        self.setup_model_with_config(config)

        parse_address = self.a_model(self.an_address_to_parse, num_workers=2)
        self.assert_properly_parse(parse_address)

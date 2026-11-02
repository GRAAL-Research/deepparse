# pylint: disable=consider-using-with

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from deepparse.network import Model2VecSeq2SeqModel
from models_evaluation.export_model_to_hub import export, repo_id_of

HIDDEN_SIZE = 16


class ExportModelToHubTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        torch.manual_seed(0)
        self.model = Model2VecSeq2SeqModel(
            input_size=256, encoder_hidden_size=HIDDEN_SIZE, decoder_hidden_size=HIDDEN_SIZE, attention_mechanism=True
        )
        self.checkpoint = self.root / "retrained_model2vec-8m_attention_address_parser.ckpt"
        torch.save(
            {
                "address_tagger_model": self.model.state_dict(),
                "model_type": "model2vec-8m_attention",
                "version": "Finetuned_",
                "seq2seq_params": {"encoder_hidden_size": HIDDEN_SIZE, "decoder_hidden_size": HIDDEN_SIZE},
            },
            self.checkpoint,
        )
        self.selected = {
            "model_name": "model2vec-8m-attention",
            "seeds": 5,
            "mean_relative_val_loss": 0.9,
            "mean_test_accuracy": 97.5,
            "std_test_accuracy": 0.3,
            "selected_run": {
                "model_type": "model2vec-8m",
                "hidden_size": HIDDEN_SIZE,
                "attention_mechanism": True,
                "seed": 44,
                "best_epoch": 28,
                "val_loss": 1.1,
                "val_accuracy": 98.1,
                "test_accuracy": 97.9,
                "run_dir": str(self.root),
                "checkpoint": str(self.checkpoint),
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_givenAModelName_whenGetItsRepository_thenReturnTheDeepparseRepository(self):
        self.assertEqual("deepparse/model2vec-8m-attention", repo_id_of("model2vec-8m-attention"))
        self.assertEqual("deepparse/model2vec-multilingual-base", repo_id_of("model2vec-multilingual"))

    def test_givenASelectedRun_whenExport_thenWriteTheHubFilesWithTheArchitecture(self):
        export_dir = export(self.selected, self.root / "export")

        self.assertEqual("model2vec-8m-attention", export_dir.name)
        config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(HIDDEN_SIZE, config["encoder_hidden_size"])
        self.assertEqual(256, config["input_size"])
        self.assertTrue(config["attention_mechanism"])
        self.assertTrue((export_dir / "model.safetensors").is_file())
        card = (export_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("seed 44, best epoch 28", card)
        self.assertIn("97.50 ± 0.30", card)

    def test_givenAnExportedModel_whenReloadLikeTheParser_thenSameWeights(self):
        export_dir = export(self.selected, self.root / "export")

        # The ModelLoader calls from_pretrained on a default model instance of the factory.
        reloaded = Model2VecSeq2SeqModel(input_size=256).from_pretrained(str(export_dir))

        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, reloaded.state_dict()[key]), key)

    def test_givenAHiddenSizeMismatch_whenExport_thenRaise(self):
        self.selected["selected_run"]["hidden_size"] = 1024

        with self.assertRaisesRegex(ValueError, "does not match"):
            export(self.selected, self.root / "export")

    def test_givenNoPushFlag_whenExport_thenNothingIsUploaded(self):
        with patch("huggingface_hub.HfApi") as hf_api:
            export(self.selected, self.root / "export")

        hf_api.assert_not_called()

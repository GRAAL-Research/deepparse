# pylint: disable=consider-using-with

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from models_evaluation.check_exported_models import build_fake_hub_cache


class CheckExportedModelsTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.export_dir = self.root / "export"
        (self.export_dir / "model2vec-8m-attention").mkdir(parents=True)
        (self.export_dir / "model2vec-8m-attention" / "config.json").write_text("{}", encoding="utf-8")
        self.embeddings_cache = self.root / "embeddings"
        (self.embeddings_cache / "models--minishlab--potion-base-8M").mkdir(parents=True)
        (self.embeddings_cache / "models--deepparse--fasttext-base").mkdir(parents=True)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_givenAnExport_whenBuildTheFakeCache_thenLayItOutLikeTheHub(self):
        build_fake_hub_cache(
            [{"model_name": "model2vec-8m-attention"}], self.export_dir, self.embeddings_cache, self.cache_dir
        )

        repo_dir = self.cache_dir / "models--deepparse--model2vec-8m-attention"
        commit = (repo_dir / "refs" / "main").read_text(encoding="utf-8")
        self.assertTrue((repo_dir / "snapshots" / commit / "config.json").is_file())
        self.assertTrue((self.cache_dir / "models--minishlab--potion-base-8M").is_symlink())
        self.assertFalse((self.cache_dir / "models--deepparse--fasttext-base").exists())

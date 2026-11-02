# pylint: disable=consider-using-with

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from deepparse.download_tools import MODEL_REPO_IDS
from models_evaluation.retrain_grid_model import (
    ResumableEarlyStopping,
    find_final_checkpoint,
    forget_model2vec_weights,
    grid_runs,
    main,
    parse_args,
    run_name,
)
from models_evaluation.select_best_model import parse_run_name


class RetrainGridModelTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_givenTheGrid_whenList_thenTwoHundredDistinctRunsSeedBySeed(self):
        runs = grid_runs()

        self.assertEqual(200, len(set(runs)))
        self.assertEqual("model2vec-8m 0 128 42", runs[0])
        self.assertTrue(all(run.endswith(" 42") for run in runs[:40]))

    def test_givenARun_whenName_thenTheSelectionParsesItBack(self):
        self.assertEqual("model2vec-8m-h256_attention-s43", run_name("model2vec-8m", 256, True, 43))
        self.assertEqual("fasttext-h1024", run_name("fasttext", 1024, False, 42))
        self.assertEqual(("model2vec-8m", 256, True, 43), parse_run_name(run_name("model2vec-8m", 256, True, 43)))

    def test_givenALogOfAResumedRun_whenTrainBegins_thenRestoreTheBestValueAndThePatienceCounter(self):
        log_path = self.root / "log.tsv"
        log_path.write_text("epoch\tval_loss\n1\t3.0\n2\t1.0\n3\t2.0\n4\t1.5\n", encoding="utf-8")
        early_stopping = ResumableEarlyStopping(log_path, monitor="val_loss", mode="min", patience=10)

        early_stopping.on_train_begin({})

        self.assertEqual((1.0, 2), (early_stopping.best, early_stopping.wait))

    def test_givenARunDirectory_whenFindTheFinalCheckpoint_thenReturnItOnlyIfSaved(self):
        self.assertIsNone(find_final_checkpoint(self.root))
        checkpoint = self.root / "retrained_model2vec-8m_address_parser.ckpt"
        checkpoint.write_bytes(b"")

        self.assertEqual(checkpoint, find_final_checkpoint(self.root))

    def test_givenTheRepositories_whenForgetModel2VecWeights_thenOnlyTheModel2VecOnesAreRemoved(self):
        with patch.dict(MODEL_REPO_IDS, clear=False):
            forget_model2vec_weights()

            self.assertFalse(any(model_type.startswith("model2vec") for model_type in MODEL_REPO_IDS))
            self.assertIn("fasttext", MODEL_REPO_IDS)

    def test_givenATestedRun_whenMain_thenSkipItWithoutBuildingAParser(self):
        run_dir = self.root / "model2vec-8m-h128-s43"
        run_dir.mkdir()
        (run_dir / "test_results.json").write_text("{}", encoding="utf-8")
        args = parse_args(
            ["model2vec-8m", "--hidden_size", "128", "--seed", "43", "--train_path", "train.p", "--test_path",
             "test.p", "--logging_dir", str(self.root), "--cache_dir", "cache"]
        )  # fmt: skip

        with patch("models_evaluation.retrain_grid_model.AddressParser") as address_parser:
            main(args)

        address_parser.assert_not_called()

    def test_givenNoModelType_whenParseArgsToTrain_thenError(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--hidden_size", "128"])

        self.assertTrue(parse_args(["--print_grid"]).print_grid)

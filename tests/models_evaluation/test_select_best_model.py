# pylint: disable=consider-using-with

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from models_evaluation.select_best_model import (
    group_runs,
    normalize_by_seed,
    parse_run_name,
    read_run,
    select_best,
    to_rst_selection_table,
    to_rst_table,
)


def make_run(checkpoints_dir: Path, run_name: str, val_losses: list, test_accuracy: float, tested: bool = True) -> None:
    run_dir = checkpoints_dir / run_name
    run_dir.mkdir()
    rows = ["epoch\ttime\tlr\tloss\taccuracy\tval_loss\tval_accuracy"]
    rows += [
        f"{epoch}\t1.0\t0.1\t1.0\t90.0\t{val_loss}\t{100 - val_loss}" for epoch, val_loss in enumerate(val_losses, 1)
    ]
    (run_dir / "log.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "retrained_model2vec-8m_address_parser.ckpt").write_bytes(b"")
    if tested:
        (run_dir / "test_results.json").write_text(json.dumps({"test_accuracy": test_accuracy}), encoding="utf-8")


class SelectBestModelTest(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.checkpoints_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def read_all(self):
        runs = [run for path in sorted(self.checkpoints_dir.iterdir()) if (run := read_run(path)) is not None]
        normalize_by_seed(runs)
        return runs

    def test_givenRunNames_whenParse_thenReturnTheConfigurationAndTheSeed(self):
        self.assertEqual(("model2vec-8m", 1024, False, 42), parse_run_name("model2vec-8m-h1024"))
        self.assertEqual(("model2vec-32m", 256, True, 45), parse_run_name("model2vec-32m-h256_attention-s45"))
        self.assertEqual(("model2vec-multilingual", 128, False, 43), parse_run_name("model2vec-multilingual-h128-s43"))
        self.assertIsNone(parse_run_name("checkpoints"))

    def test_givenAnUntestedRun_whenRead_thenIgnoreIt(self):
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [3.0, 2.0], 97.0, tested=False)

        self.assertEqual([], self.read_all())

    def test_givenARun_whenRead_thenUseTheEpochWithTheLowestValidationLoss(self):
        make_run(self.checkpoints_dir, "model2vec-8m-h128-s43", [3.0, 1.5, 2.0], 97.0)

        run = self.read_all()[0]

        self.assertEqual(
            (2, 1.5, 98.5, 97.0, 43), (run.best_epoch, run.val_loss, run.val_accuracy, run.test_accuracy, run.seed)
        )

    def test_givenAGrid_whenSelect_thenKeepTheHiddenSizeWithTheLowestMeanValidationLossThenTheBestSeed(self):
        # Seed 42 mean: 1.533, seed 43 mean: 2.167. h128 has the best single run (1.0) but the worst mean relative
        # validation loss (1.018 against 0.845 for h256).
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [1.0], 99.9)
        make_run(self.checkpoints_dir, "model2vec-8m-h128-s43", [3.0], 95.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256", [1.6], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256-s43", [1.4], 96.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256_attention", [2.0], 96.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256_attention-s43", [2.1], 96.5)

        selection = select_best(group_runs(self.read_all()), expected_seeds=2)

        configuration, run = selection[("model2vec-8m", False)]
        self.assertEqual((256, 43), (configuration.hidden_size, run.seed))
        # 2.1 is a higher validation loss than 2.0, but relative to its seed (0.969 against 1.304), it is the best.
        configuration, run = selection[("model2vec-8m", True)]
        self.assertEqual((256, 43), (configuration.hidden_size, run.seed))

    def test_givenASeedWithAnEasierValidationSplit_whenSelect_thenDoNotPreferThisSeed(self):
        # Seed 46 halves every validation loss (easier split), but its h128 run is the worst relative to its seed.
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [1.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256", [2.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h128-s46", [0.6], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256-s46", [0.5], 97.0)

        configuration, run = select_best(group_runs(self.read_all()), expected_seeds=2)[("model2vec-8m", False)]

        self.assertEqual((128, 42), (configuration.hidden_size, run.seed))
        self.assertAlmostEqual(1.0 / 1.5, run.relative_val_loss)

    def test_givenAnIncompleteConfiguration_whenSelect_thenRaiseUnlessAllowed(self):
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [1.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h128-s43", [1.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256", [1.0], 97.0)
        configurations = group_runs(self.read_all())

        with self.assertRaisesRegex(ValueError, "model2vec-8m-h256 \\(1 runs\\)"):
            select_best(configurations, expected_seeds=2)
        self.assertIn(("model2vec-8m", False), select_best(configurations, expected_seeds=2, allow_incomplete=True))

    def test_givenASelection_whenRenderTheTable_thenTheSelectedConfigurationIsInBold(self):
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [2.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256", [1.0], 98.0)
        configurations = group_runs(self.read_all())

        table = to_rst_table(configurations, select_best(configurations, expected_seeds=1))

        self.assertIn("            - 128\n", table)
        self.assertIn("            - **256**\n", table)
        self.assertIn("**98.00 ± 0.00**", table)

    def test_givenASelection_whenRenderTheSelectionTable_thenOneRowPerReleasedModel(self):
        make_run(self.checkpoints_dir, "model2vec-8m-h128", [2.0], 97.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256", [1.0], 98.0)
        make_run(self.checkpoints_dir, "model2vec-8m-h256_attention", [1.5], 96.5)

        table = to_rst_selection_table(select_best(group_runs(self.read_all()), expected_seeds=1))

        self.assertIn("        *   - ``model2vec-8m``\n            - 256\n            - 98.00\n", table)
        self.assertIn("``model2vec-8m-attention``", table)
        self.assertEqual(3, table.count("        *   - "))

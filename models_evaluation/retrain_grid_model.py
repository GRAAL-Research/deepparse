"""
Train one run of the grid of the pretrained models (Model2Vec, FastText or BPEmb) from scratch and test it.

The October 2026 grid trains each model type with four hidden sizes (128, 256, 512 and 1024), with and without
attention, with five seeds (42 to 46), on the deepparse split of the worldwide-addresses data (``train.p`` and
``test.p``). The training uses 90% of ``train.p`` (the seed also draws this split), batches of 2,048 addresses
(1,024 for BPEmb, to fit in the GPU memory), a learning rate of 0.1 reduced by 10 after 3 epochs without improvement
of the validation loss, and an early stopping after 10 epochs without improvement, for at most 100 epochs. The released
models are then selected with ``select_best_model.py`` and exported with ``export_model_to_hub.py``.

Each run writes ``<logging_dir>/<model_type>-h<hidden_size>[_attention][-s<seed>]`` (no seed suffix for the seed 42)
with the Poutyne logs and checkpoints, the ``retrained_*_address_parser.ckpt`` file (weights of the best epoch) and
``test_results.json``. The training is resumable: rerunning the same command with the same ``--logging_dir`` continues
an interrupted run (e.g. a Slurm job killed by its time limit). Poutyne restores the weights, the optimizer (including
the current learning rate), the random states and the best checkpoint tracking; this script also restores the
ReduceLROnPlateau state and the EarlyStopping counters. A finished run (final checkpoint saved) is only tested, and a
tested run is skipped.

Usage:
    python -m models_evaluation.retrain_grid_model --print_grid > queue.txt  # One run per line.
    python -m models_evaluation.retrain_grid_model model2vec-8m --attention --hidden_size 256 --seed 43 \\
        --train_path data/train.p --test_path data/test.p --logging_dir checkpoints --cache_dir cache --device 0
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from poutyne import EarlyStopping, LRSchedulerCheckpoint, ReduceLROnPlateau

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.download_tools import MODEL_REPO_IDS
from deepparse.parser import AddressParser

GRID_MODEL_TYPES = ("model2vec-8m", "model2vec-32m", "model2vec-multilingual", "fasttext", "bpemb")
GRID_HIDDEN_SIZES = (128, 256, 512, 1024)
GRID_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_SEED = 42
LR_SCHEDULER_STATE_FILENAME = "reduce_lr_on_plateau.ckpt"


class ResumableEarlyStopping(EarlyStopping):
    """
    EarlyStopping that restores its best value and patience counter from the Poutyne log of a resumed run.
    """

    def __init__(self, log_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_path = log_path

    def on_train_begin(self, logs: dict) -> None:
        super().on_train_begin(logs)
        if not self.log_path.is_file():
            return
        with open(self.log_path, encoding="utf-8") as file:
            values = [float(row[self.monitor]) for row in csv.DictReader(file, delimiter="\t")]
        if values:
            best_index = int(np.argmin(values)) if self.mode == "min" else int(np.argmax(values))
            self.best = values[best_index]
            self.wait = len(values) - 1 - best_index
            print(f"Resumed EarlyStopping: best {self.monitor}={self.best:.6f}, wait={self.wait}", flush=True)


def grid_runs() -> List[str]:
    """
    The runs of the grid, one ``"<model_type> <attention 0/1> <hidden_size> <seed>"`` line per run, ordered so that
    the seeds are trained one after the other (a partial grid then has whole seeds).
    """
    return [
        f"{model_type} {attention} {hidden_size} {seed}"
        for seed in GRID_SEEDS
        for hidden_size in GRID_HIDDEN_SIZES
        for model_type in GRID_MODEL_TYPES
        for attention in (0, 1)
    ]


def run_name(model_type: str, hidden_size: int, attention: bool, seed: int) -> str:
    name = f"{model_type}-h{hidden_size}{'_attention' if attention else ''}"
    # The first runs used the default seed without suffix: keep their directory names.
    return name if seed == DEFAULT_SEED else f"{name}-s{seed}"


def forget_model2vec_weights() -> None:
    """
    Make the parser build a randomly initialized Model2Vec network instead of downloading the published weights, as
    when the grid was trained (before the weights were published). The training starts from scratch anyway
    (``seq2seq_params``), and this keeps the random number generator path, hence the runs, reproducible.
    """
    for model_type in [model_type for model_type in MODEL_REPO_IDS if model_type.startswith("model2vec")]:
        del MODEL_REPO_IDS[model_type]


def find_final_checkpoint(logging_path: Path) -> Path | None:
    checkpoints = sorted(logging_path.glob("retrained_*_address_parser.ckpt"))
    return checkpoints[0] if checkpoints else None


def main(args: argparse.Namespace) -> None:
    logging_path = Path(args.logging_dir) / run_name(args.model_type, args.hidden_size, args.attention, args.seed)
    if (logging_path / "test_results.json").is_file():
        print(f"{logging_path.name}: already tested, skipped", flush=True)
        return

    forget_model2vec_weights()
    final_checkpoint = find_final_checkpoint(logging_path)
    if final_checkpoint is None:
        # The random init of the model happens before retrain seeds the training, so we seed it too.
        torch.manual_seed(args.seed)
        address_parser = AddressParser(
            model_type=args.model_type,
            attention_mechanism=args.attention,
            device=args.device,
            cache_dir=args.cache_dir,
            offline=True,
        )

        reduce_lr = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=args.lr_patience, factor=0.1)
        lr_scheduler_state = logging_path / LR_SCHEDULER_STATE_FILENAME
        if lr_scheduler_state.is_file():
            # Applied by the scheduler when the training begins, once the optimizer exists.
            reduce_lr.load_state(str(lr_scheduler_state))
            print(f"Resumed ReduceLROnPlateau from {lr_scheduler_state}", flush=True)
        callbacks = [
            ResumableEarlyStopping(logging_path / "log.tsv", monitor="val_loss", mode="min", patience=args.patience),
            # It drives reduce_lr (which must not be passed itself, or it would step twice per epoch) and saves its
            # state after each epoch.
            LRSchedulerCheckpoint(reduce_lr, str(lr_scheduler_state), verbose=False),
        ]

        address_parser.retrain(
            PickleDatasetContainer(args.train_path),
            train_ratio=0.9,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            callbacks=callbacks,
            seed=args.seed,
            logging_path=str(logging_path),
            name_of_the_retrain_parser=None,
            seq2seq_params={"encoder_hidden_size": args.hidden_size, "decoder_hidden_size": args.hidden_size},
        )
    else:
        print(f"{logging_path.name}: training already finished, testing {final_checkpoint}", flush=True)
        address_parser = AddressParser(
            model_type=args.model_type,
            attention_mechanism=args.attention,
            device=args.device,
            cache_dir=args.cache_dir,
            offline=True,
            path_to_retrained_model=str(final_checkpoint),
        )

    test_container = PickleDatasetContainer(args.test_path, is_training_container=True)
    test_results = address_parser.test(test_container, batch_size=args.batch_size, num_workers=args.num_workers)
    (logging_path / "test_results.json").write_text(json.dumps(test_results, indent=2), encoding="utf-8")
    print(logging_path.name, test_results, flush=True)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_type", type=str, nargs="?")
    parser.add_argument("--print_grid", action="store_true", help="Print the runs of the grid, one per line, and exit.")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--logging_dir", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048, help="2,048 in the grid, 1,024 for BPEmb.")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # A patience of 5 (2 for the learning rate) stopped some runs too early (15-20 epochs, much worse validation loss),
    # hence a high variance between the seeds.
    parser.add_argument("--patience", type=int, default=10, help="EarlyStopping patience (epochs).")
    parser.add_argument("--lr_patience", type=int, default=3, help="ReduceLROnPlateau patience (epochs).")
    args = parser.parse_args(argv)
    if not args.print_grid:
        missing = [
            name
            for name in ("model_type", "train_path", "test_path", "logging_dir", "cache_dir")
            if not getattr(args, name)
        ]
        if missing:
            parser.error(f"the following arguments are required to train: {', '.join(missing)}")
    if args.device.isdigit():
        args.device = int(args.device)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.print_grid:
        print("\n".join(grid_runs()))
    else:
        main(parsed_args)

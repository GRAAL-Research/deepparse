"""
Select the configuration and the seed of each pretrained model to release from a grid of retraining runs.

Each run directory of the grid is named ``<model_type>-h<hidden_size>[_attention][-s<seed>]`` (the runs with the
default seed 42 have no seed suffix) and holds the Poutyne ``log.tsv``, the ``test_results.json`` written after the
test and the ``retrained_*_address_parser.ckpt`` file (the weights of the best epoch, according to the validation
loss).

The selection only uses the validation split (``val_loss`` of the best epoch, the metric monitored by the early
stopping). The seed of a run also draws its train/validation split (``train_ratio``), so some seeds have an easier
validation split than others (in the October 2026 grid, the validation loss of the seed 46 runs is about half the
others, for the same test accuracy). We thus divide the validation loss of each run by the mean validation loss of the
runs with the same seed (its relative validation loss). For each model type, with and without attention, we keep the
hidden size with the lowest mean relative validation loss over the seeds, then the seed with the lowest relative
validation loss for this hidden size. The test accuracy is only reported.

Usage:
    python -m models_evaluation.select_best_model checkpoints_p10 --output selection.json \\
        --rst_table docs/source/model2vec_grid.inc --rst_selection_table docs/source/model2vec_selection.inc
"""

import argparse
import csv
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

MODEL2VEC_MODEL_TYPES = ("model2vec-8m", "model2vec-32m", "model2vec-multilingual")
DEFAULT_SEED = 42

RUN_NAME_PATTERN = re.compile(
    r"^(?P<model_type>.+)-h(?P<hidden_size>\d+)(?P<attention>_attention)?(?:-s(?P<seed>\d+))?$"
)


@dataclass
class Run:
    model_type: str
    hidden_size: int
    attention_mechanism: bool
    seed: int
    best_epoch: int
    val_loss: float
    val_accuracy: float
    test_accuracy: float
    run_dir: str
    checkpoint: str
    # The validation loss divided by the mean validation loss of the runs with the same seed (see normalize_by_seed).
    relative_val_loss: float = 1.0


@dataclass
class Configuration:
    model_type: str
    hidden_size: int
    attention_mechanism: bool
    runs: List[Run]

    @property
    def mean_relative_val_loss(self) -> float:
        return statistics.mean(run.relative_val_loss for run in self.runs)

    def summary(self, metric: str) -> Tuple[float, float]:
        values = [getattr(run, metric) for run in self.runs]
        return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def parse_run_name(run_name: str) -> Tuple[str, int, bool, int] | None:
    """
    Parse a run directory name into ``(model_type, hidden_size, attention_mechanism, seed)``, or ``None`` if the name
    does not follow the grid convention.
    """
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return None
    seed = int(match["seed"]) if match["seed"] is not None else DEFAULT_SEED
    return match["model_type"], int(match["hidden_size"]), match["attention"] is not None, seed


def read_run(run_dir: Path) -> Run | None:
    """
    Read a finished run (tested, with its final checkpoint), or return ``None`` if the run is not finished.
    """
    parsed_name = parse_run_name(run_dir.name)
    test_results_path = run_dir / "test_results.json"
    log_path = run_dir / "log.tsv"
    checkpoints = sorted(run_dir.glob("retrained_*_address_parser.ckpt"))
    if parsed_name is None or not test_results_path.is_file() or not log_path.is_file() or not checkpoints:
        return None

    with open(log_path, encoding="utf-8") as file:
        epochs = list(csv.DictReader(file, delimiter="\t"))
    best_epoch = min(epochs, key=lambda epoch: float(epoch["val_loss"]))
    test_results = json.loads(test_results_path.read_text(encoding="utf-8"))

    model_type, hidden_size, attention_mechanism, seed = parsed_name
    return Run(
        model_type=model_type,
        hidden_size=hidden_size,
        attention_mechanism=attention_mechanism,
        seed=seed,
        best_epoch=int(best_epoch["epoch"]),
        val_loss=float(best_epoch["val_loss"]),
        val_accuracy=float(best_epoch["val_accuracy"]),
        test_accuracy=float(test_results["test_accuracy"]),
        run_dir=str(run_dir),
        checkpoint=str(checkpoints[0]),
    )


def normalize_by_seed(runs: List[Run]) -> None:
    """
    Set the relative validation loss of each run: its validation loss divided by the mean validation loss of the runs
    with the same seed, since the seed also draws the validation split.
    """
    seed_val_losses: Dict[int, List[float]] = {}
    for run in runs:
        seed_val_losses.setdefault(run.seed, []).append(run.val_loss)
    for run in runs:
        run.relative_val_loss = run.val_loss / statistics.mean(seed_val_losses[run.seed])


def group_runs(runs: List[Run]) -> List[Configuration]:
    configurations: Dict[Tuple[str, int, bool], Configuration] = {}
    for run in runs:
        key = (run.model_type, run.hidden_size, run.attention_mechanism)
        configurations.setdefault(key, Configuration(*key, runs=[])).runs.append(run)
    for configuration in configurations.values():
        configuration.runs.sort(key=lambda run: run.seed)
    return sorted(configurations.values(), key=lambda c: (c.model_type, c.attention_mechanism, c.hidden_size))


def select_best(
    configurations: List[Configuration], expected_seeds: int, allow_incomplete: bool = False
) -> Dict[Tuple[str, bool], Tuple[Configuration, Run]]:
    """
    For each ``(model_type, attention_mechanism)``, select the hidden size with the lowest mean relative validation
    loss, then the seed with the lowest relative validation loss for this hidden size (see :func:`normalize_by_seed`).

    Raises:
        ValueError: if a configuration does not have ``expected_seeds`` finished runs and ``allow_incomplete`` is
            ``False``, since the comparison between the hidden sizes would then be unfair.
    """
    incomplete = [c for c in configurations if len(c.runs) != expected_seeds]
    if incomplete and not allow_incomplete:
        details = ", ".join(
            f"{c.model_type}-h{c.hidden_size}{'_attention' if c.attention_mechanism else ''} ({len(c.runs)} runs)"
            for c in incomplete
        )
        raise ValueError(f"Some configurations do not have {expected_seeds} finished runs: {details}.")

    selection = {}
    for configuration in configurations:
        key = (configuration.model_type, configuration.attention_mechanism)
        if key not in selection or configuration.mean_relative_val_loss < selection[key][0].mean_relative_val_loss:
            selection[key] = (configuration, None)
    return {
        key: (configuration, min(configuration.runs, key=lambda run: run.relative_val_loss))
        for key, (configuration, _) in selection.items()
    }


def model_name(model_type: str, attention_mechanism: bool) -> str:
    return f"{model_type}-attention" if attention_mechanism else model_type


def to_rst_table(
    configurations: List[Configuration], selection: Dict[Tuple[str, bool], Tuple[Configuration, Run]]
) -> str:
    """
    Render every configuration of the grid as a reStructuredText list-table; the selected configurations are in bold.
    """
    lines = [
        ".. list-table::",
        "        :header-rows: 1",
        "",
        "        *   - Model",
        "            - Hidden size",
        "            - Seeds",
        "            - Relative validation loss",
        "            - Validation loss",
        "            - Validation accuracy (%)",
        "            - Test accuracy (%)",
    ]
    for configuration in configurations:
        selected = selection[(configuration.model_type, configuration.attention_mechanism)][0] is configuration
        cells = [
            f"``{model_name(configuration.model_type, configuration.attention_mechanism)}``",
            str(configuration.hidden_size),
            str(len(configuration.runs)),
        ]
        for metric, precision in (("relative_val_loss", 3), ("val_loss", 3), ("val_accuracy", 2), ("test_accuracy", 2)):
            mean, std = configuration.summary(metric)
            cells.append(f"{mean:.{precision}f} ± {std:.{precision}f}")
        if selected:
            cells = [cells[0]] + [f"**{cell}**" for cell in cells[1:]]
        lines.append(f"        *   - {cells[0]}")
        lines.extend(f"            - {cell}" for cell in cells[1:])
    return "\n".join(lines) + "\n"


def to_rst_selection_table(selection: Dict[Tuple[str, bool], Tuple[Configuration, Run]]) -> str:
    """
    Render the released models (one per model type, with and without attention) as a reStructuredText list-table.
    """
    lines = [
        ".. list-table::",
        "        :header-rows: 1",
        "",
        "        *   - Model",
        "            - Hidden size",
        "            - Test accuracy of the released model (%)",
        "            - Test accuracy, mean ± std over the seeds (%)",
    ]
    for (model_type, attention_mechanism), (configuration, run) in sorted(selection.items()):
        mean, std = configuration.summary("test_accuracy")
        lines.append(f"        *   - ``{model_name(model_type, attention_mechanism)}``")
        lines.append(f"            - {configuration.hidden_size}")
        lines.append(f"            - {run.test_accuracy:.2f}")
        lines.append(f"            - {mean:.2f} ± {std:.2f} ({len(configuration.runs)} seeds)")
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> None:
    runs = [run for run_dir in sorted(Path(args.checkpoints_dir).iterdir()) if (run := read_run(run_dir)) is not None]
    runs = [run for run in runs if run.model_type in args.model_types]
    normalize_by_seed(runs)
    configurations = group_runs(runs)
    selection = select_best(configurations, args.expected_seeds, allow_incomplete=args.allow_incomplete)

    output = []
    for (model_type, attention_mechanism), (configuration, run) in sorted(selection.items()):
        test_mean, test_std = configuration.summary("test_accuracy")
        output.append(
            {
                "model_name": model_name(model_type, attention_mechanism),
                "seeds": len(configuration.runs),
                "mean_relative_val_loss": configuration.mean_relative_val_loss,
                "mean_test_accuracy": test_mean,
                "std_test_accuracy": test_std,
                "selected_run": asdict(run),
            }
        )
        print(
            f"{model_name(model_type, attention_mechanism):35s} h{configuration.hidden_size:<5d} seed {run.seed} "
            f"(epoch {run.best_epoch}, relative val_loss {run.relative_val_loss:.3f}, test {run.test_accuracy:.2f}; "
            f"{len(configuration.runs)} seeds: test {test_mean:.2f} ± {test_std:.2f})"
        )

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.rst_table is not None:
        Path(args.rst_table).write_text(to_rst_table(configurations, selection), encoding="utf-8")
    if args.rst_selection_table is not None:
        Path(args.rst_selection_table).write_text(to_rst_selection_table(selection), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints_dir", type=str, help="The directory holding one sub-directory per run.")
    parser.add_argument("--output", type=str, default="selection.json", help="The selection JSON file to write.")
    parser.add_argument("--rst_table", type=str, default=None, help="Also write the grid as a reST table.")
    parser.add_argument(
        "--rst_selection_table", type=str, default=None, help="Also write the released models as a reST table."
    )
    parser.add_argument("--model_types", nargs="+", default=list(MODEL2VEC_MODEL_TYPES))
    parser.add_argument("--expected_seeds", type=int, default=5)
    parser.add_argument(
        "--allow_incomplete", action="store_true", help="Select even if some configurations miss runs (preview only)."
    )
    main(parser.parse_args())

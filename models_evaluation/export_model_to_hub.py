"""
Export the selected retrained models to the Hugging Face Hub format of the pretrained models (``config.json``,
``model.safetensors`` and a model card), check that the export gives the same weights, and optionally push it.

Nothing is pushed without ``--push``.

Usage:
    python -m models_evaluation.export_model_to_hub selection.json --output_dir hub_export
    python -m models_evaluation.export_model_to_hub selection.json --output_dir hub_export --push
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from deepparse.download_tools import MODEL2VEC_MODELS, MODEL_MAPPING_CHOICES, MODEL_REPO_IDS, get_model2vec_model_type
from deepparse.network import Model2VecSeq2SeqModel
from deepparse.weights_tools import handle_weights_upload

OUTPUT_SIZE = 9  # The 8 deepparse tags and the EOS tag.

MODEL_CARD_TEMPLATE = """---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- address-parsing
- deepparse
license: mit
datasets:
- deepparse/worldwide-addresses
library_name: deepparse
---

# {title}

A [deepparse](https://deepparse.org) address parser (Seq2Seq, {architecture}) using the
[{embeddings_repo}](https://huggingface.co/{embeddings_repo}) Model2Vec static embeddings ({embeddings_dim}
dimensions).

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model_type="{model_type}"{attention_argument})
address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")
```

Install it with `pip install deepparse[model2vec]`.

## Training

- Data: the [deepparse/worldwide-addresses](https://huggingface.co/datasets/deepparse/worldwide-addresses) dataset
  (20 countries, 100,000 clean and 50,000 incomplete addresses per country), 90% for training and 10% for validation.
  `Suburb` and `District` are mapped to `Municipality`, and the `County` and `Country` words are removed.
- Architecture: encoder and decoder hidden size {hidden_size}, one layer each{attention_sentence}.
- Selection: the hidden size (128, 256, 512 or 1024) with the lowest mean validation loss over {seeds} seeds, then the
  seed with the lowest validation loss (seed {seed}, best epoch {best_epoch}).

## Results

| | Accuracy (%) |
|---|---|
| Test, this model | {test_accuracy:.2f} |
| Test, mean ± std over the {seeds} seeds of this configuration | {mean_test_accuracy:.2f} ± {std_test_accuracy:.2f} |
| Validation, this model | {val_accuracy:.2f} |

The accuracy is the per-word tag accuracy on the test split of the rebuilt worldwide-addresses data.

## Limitations

The source data has no `Orientation` nor `GeneralDelivery` tags: the model has these outputs but never predicts them
(the street orientation, e.g. `Ouest` or `NW`, is part of the `StreetName`). South Korea has almost only incomplete
addresses.
"""


def repo_id_of(model_name: str) -> str:
    return MODEL_REPO_IDS[MODEL_MAPPING_CHOICES[model_name]]


def load_selected_model(selected_run: Dict) -> Model2VecSeq2SeqModel:
    checkpoint = handle_weights_upload(path_to_model_to_upload=selected_run["checkpoint"], device=torch.device("cpu"))
    seq2seq_params = checkpoint.get("seq2seq_params") or {}
    if seq2seq_params.get("encoder_hidden_size", 1024) != selected_run["hidden_size"]:
        raise ValueError(f"The checkpoint {selected_run['checkpoint']} does not match the selected hidden size.")

    model = Model2VecSeq2SeqModel(
        input_size=MODEL2VEC_MODELS[get_model2vec_model_type(selected_run["model_type"])][1],
        output_size=OUTPUT_SIZE,
        attention_mechanism=selected_run["attention_mechanism"],
        **seq2seq_params,
    )
    model.load_state_dict(checkpoint["address_tagger_model"])
    return model


def write_model_card(export_dir: Path, selected: Dict) -> None:
    run = selected["selected_run"]
    embeddings_repo, embeddings_dim = MODEL2VEC_MODELS[get_model2vec_model_type(run["model_type"])]
    attention = run["attention_mechanism"]
    card = MODEL_CARD_TEMPLATE.format(
        title=f"deepparse {selected['model_name']}",
        architecture="with attention mechanism" if attention else "without attention mechanism",
        embeddings_repo=embeddings_repo,
        embeddings_dim=embeddings_dim,
        model_type=run["model_type"],
        attention_argument=", attention_mechanism=True" if attention else "",
        hidden_size=run["hidden_size"],
        attention_sentence=", with an attention mechanism" if attention else "",
        seeds=selected["seeds"],
        seed=run["seed"],
        best_epoch=run["best_epoch"],
        test_accuracy=run["test_accuracy"],
        mean_test_accuracy=selected["mean_test_accuracy"],
        std_test_accuracy=selected["std_test_accuracy"],
        val_accuracy=run["val_accuracy"],
    )
    (export_dir / "README.md").write_text(card, encoding="utf-8")


def export(selected: Dict, output_dir: Path) -> Path:
    """
    Export one selected model to ``output_dir/<repository name>`` and check that reloading it with
    ``from_pretrained`` (as :class:`~deepparse.network.ModelLoader` does) gives the same weights.
    """
    model = load_selected_model(selected["selected_run"])
    export_dir = output_dir / repo_id_of(selected["model_name"]).split("/")[1]
    model.save_pretrained(export_dir)
    write_model_card(export_dir, selected)

    reloaded = Model2VecSeq2SeqModel.from_pretrained(str(export_dir))
    expected, actual = model.state_dict(), reloaded.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(expected[key], actual[key]) for key in expected):
        raise RuntimeError(f"The exported model {export_dir} does not reload with the same weights.")
    return export_dir


def push(export_dir: Path, repo_id: str, commit_message: str) -> None:
    # pylint: disable=import-outside-toplevel
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(export_dir), repo_id=repo_id, commit_message=commit_message)


def main(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for selected in selection:
        export_dir = export(selected, output_dir)
        repo_id = repo_id_of(selected["model_name"])
        print(f"{selected['model_name']}: exported to {export_dir} (repository {repo_id})")
        if args.push:
            push(export_dir, repo_id, args.commit_message)
            print(f"{selected['model_name']}: pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selection", type=str, help="The selection JSON file written by select_best_model.")
    parser.add_argument("--output_dir", type=str, default="hub_export")
    parser.add_argument("--push", action="store_true", help="Push the exported models to the Hugging Face Hub.")
    parser.add_argument("--commit_message", type=str, default="Add the pretrained weights")
    main(parser.parse_args())

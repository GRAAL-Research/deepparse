---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- address-parsing
- deepparse
license: mit
library_name: deepparse
---

# deepparse FastText

A [deepparse](https://deepparse.org) address parser (Seq2Seq, without attention mechanism) using the French [fastText](https://fasttext.cc/) Common Crawl word embeddings (`cc.fr.300.bin`, with character n-grams).

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model_type="fasttext")
address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")
```

The `fasttext-light` model (`model_type="fasttext-light"`) uses the same weights with the
[Magnitude Light](https://github.com/davebulaval/magnitude-light) embeddings, stored on the `light-embeddings`
revision of this repository.

## Architecture

One-layer LSTM encoder and decoder, hidden size 1,024, fastText embeddings of 300 dimensions. The
model predicts 8 tags (`StreetNumber`, `StreetName`, `Unit`, `Municipality`, `Province`, `PostalCode`, `Orientation`
and `GeneralDelivery`) and the end-of-sequence tag.

## Training

Trained in 2020 on the dataset described in the
[deepparse-address-data](https://github.com/GRAAL-Research/deepparse-address-data) repository (addresses of 20
countries, built from the [libpostal](https://github.com/openvenues/libpostal) data). See
[Leveraging Subword Embeddings for Multinational Address Parsing](https://arxiv.org/abs/2006.16152) and
[Multinational Address Parsing: A Zero-Shot Evaluation](https://arxiv.org/abs/2112.04008).

## Results

- 20 training countries, clean addresses: 99.51% mean per-country accuracy (98.90% to 99.97%).
- 20 training countries, incomplete addresses: 98.42% (91.10% to 99.55%).
- 41 zero-shot countries: 83.46% (36.26% to 96.02%).

Per-country tables: the [Results page](https://deepparse.org/results.html) of the documentation.

## Limitations

- On the test split of the newer
  [worldwide-addresses](https://huggingface.co/datasets/deepparse/worldwide-addresses) data (same 20 countries,
  clean and incomplete addresses, 260,000 addresses), this model reaches 89.35% per-word accuracy (measured in
  November 2026). It is weaker on incomplete addresses (e.g. a postal code or a municipality alone) than on complete
  ones.
- In practice, it rarely predicts the `Orientation` and `GeneralDelivery` tags: in November 2026, it predicted
  neither on 12,000 addresses of the worldwide-addresses test split nor on typical cases such as `Ouest`, `NW` or
  `PO Box`.

## Citation

```bibtex
@misc{yassine2020leveraging,
    title={{Leveraging Subword Embeddings for Multinational Address Parsing}},
    author={Marouane Yassine and David Beauchemin and François Laviolette and Luc Lamontagne},
    year={2020},
    eprint={2006.16152},
    archivePrefix={arXiv}
}
```

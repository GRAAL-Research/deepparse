<img src="https://raw.githubusercontent.com/GRAAL-Research/deepparse/master/docs/source/_static/logos/logo.png" width="150" height="135"/>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deepparse)](https://pypi.org/project/deepparse)
[![PyPI Status](https://badge.fury.io/py/deepparse.svg)](https://badge.fury.io/py/deepparse)
[![PyPI Status](https://pepy.tech/badge/deepparse)](https://pepy.tech/project/deepparse)
[![Continuous Integration](https://github.com/GRAAL-Research/deepparse/workflows/Continuous%20Integration/badge.svg)](https://github.com/GRAAL-Research/deepparse/actions?query=workflow%3A%22Continuous+Integration%22+branch%3Amaster)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)

## Here is deepparse.

Deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepparse to:

- Use the pre-trained models to parse multinational addresses.
- Retrain our pre-trained models on new data to parse multinational addresses.

Read the documentation at [deepparse.org](https://deepparse.org).

Deepparse is compatible with the __latest version of PyTorch__ and  __Python >= 3.6__.

### Countries and Results

The following table presents the accuracy on the 20 countries we used during training for both our models.

| Country     |   Fasttext (%) |   BPEmb (%) | Country        |   Fasttext (%) |   BPEmb (%) |
|:------------|---------------:|------------:|:---------------|---------------:|------------:|
| Italy       |          99.66 |       99.73 | United States  |          99.56 |       99.53 |
| Germany     |          99.72 |       99.84 | Austria        |          99.19 |       99.03 |
| South Korea |          99.96 |       100.00| Canada         |          99.76 |       99.80 |
| Mexico      |          99.54 |       99.60 | Australia      |          99.62 |       99.74 |
| Finland     |          99.75 |       99.87 | Netherlands    |          99.50 |       99.84 |
| France      |          99.54 |       99.50 | United Kingdom |          99.54 |       99.62 |
| Russia      |          98.71 |       99.49 | Norway         |          99.40 |       98.71 |
| Switzerland |          99.48 |       99.61 | Poland         |          99.64 |       99.83 |
| Brazil      |          99.33 |       99.24 | Denmark        |          99.65 |       99.84 |
| Spain       |          99.70 |       99.79 | Czechia        |          99.46 |       99.83 |


We have also made a zero-shot evaluation of our models using data from 41 other countries; the results are shown in the next table.

| Country       |   Fasttext (%) |   BPEmb (%) | Country       |   Fasttext (%) |   BPEmb (%) |
|:--------------|---------------:|------------:|:--------------|---------------:|------------:|
| Philippines   |          81.56 |       83.73 | South Africa  |          92.69 |       95.03 |
| Colombia      |          85.92 |       87.50 | Venezuela     |          95.36 |       89.67 |
| Bermuda       |          91.30 |       93.66 | Lithuania     |          89.21 |       76.60 |
| Moldova       |          88.51 |       89.13 | India         |          66.91 |       77.26 |
| Malaysia      |          81.31 |       92.78 | Bosnia        |          88.91 |       84.33 |
| Belgium       |          89.57 |       86.41 | Ukraine       |          91.80 |       92.73 |
| Greece        |          83.42 |       39.82 | Algeria       |          86.93 |       80.62 |
| Slovakia      |          81.00 |       91.28 | Bangladesh    |          74.49 |       79.29 |
| Latvia        |          93.80 |       80.18 | Reunion       |          96.48 |       93.40 |
| Romania       |          93.23 |       91.83 | Singapore     |          84.55 |       81.68 |
| Indonesia     |          63.15 |       67.97 | Cyprus        |          97.69 |       98.30 |
| Portugal      |          93.39 |       93.20 | Serbia        |          95.62 |       94.69 |
| Croatia       |          96.63 |       86.24 | Japan         |          44.33 |       35.77 |
| New Caledonia |          99.42 |       99.01 | New Zealand   |          97.04 |       98.86 |
| Uzbekistan    |          87.63 |       71.93 | Faroe Islands |          71.73 |       85.46 |
| Hungary       |          47.00 |       24.05 | Slovenia      |          96.27 |       97.28 |
| Paraguay      |          97.00 |       97.15 | Iceland       |          95.76 |       98.01 |
| Estonia       |          90.61 |       76.45 | Argentina     |          89.47 |       88.55 |
| Bulgaria      |          92.70 |       95.87 | Sweden        |          77.29 |       87.77 |
| Belarus       |          88.77 |       93.00 | Kazakhstan    |          87.24 |       91.23 |
| Ireland       |          86.35 |       87.49 |

#### Noisy data
The following table presents the accuracy on the 20 countries we used during training for both our models but for noisy 
and incomplete data. We didn't test on the other 41 countries since we did not train on them and the results would most likely be even lower.

> We are working on better models for noisy data which should be made available during January 2021. 

| Country     |   Fasttext (%) |   BPEmb (%) | Country        |   Fasttext (%) |   BPEmb (%) |
|:------------|---------------:|------------:|:---------------|---------------:|------------:|
| Italy       |          64.36 |       71.18 | United States  |          57.59 |       58.35 |
| Germany     |          39.63 |       52.46 | Austria        |          61.93 |       73.64 |
| South Korea |          27.67 |       40.93 | Canada         |          70.42 |       70.31 |
| Mexico      |          50.68 |       43.81 | Australia      |          72.38 |       72.92 |
| Finland     |          43.23 |       68.26 | Netherlands    |          47.42 |       69.62 |
| France      |          65.3  |       66.93 | United Kingdom |          44.82 |       43.03 |
| Russia      |          52.75 |       47.48 | Norway         |          54.37 |       71.26 |
| Switzerland |          54.13 |       66.27 | Poland         |          37.89 |       43.31 |
| Brazil      |          48.07 |       46.25 | Denmark        |          48.12 |       56.76 |
| Spain       |          68.19 |       77.27 | Czechia        |          52.86 |       62.88 |


## Getting started:

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model_type="bpemb", device=0)

# you can parse one address
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

# or multiple addresses
parsed_address = address_parser(
    ["350 rue des Lilas Ouest Québec Québec G1L 1B6", "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

# you can also get the probability of the predicted tags
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6", with_prob=True)
```

The predictions tags are the following

- "StreetNumber": for the street number
- "StreetName": for the name of the street
- "Unit": for the unit (such as apartment)
- "Municipality": for the municipality
- "Province": for the province or local region
- "PostalCode": for the postal code
- "Orientation": for the street orientation (e.g. west, east)
- "GeneralDelivery": for other delivery information

### Retrain a model
> see [here](https://github.com/GRAAL-Research/deepparse/blob/master/examples/fine_tuning.py) for a complete example.

```python
# We will retrain the fasttext version of our pretrained model.
address_parser = AddressParser(model_type="fasttext", device=0)

address_parser.retrain(training_container, 0.8, epochs=5, batch_size=8)

```

------------------

## Installation

Before installing deepparse, you must have the latest version of [PyTorch](https://pytorch.org/) in your environment.

- **Install the stable version of deepparse:**

```sh
pip install deepparse
```

- **Install the latest development version of deepparse:**

```sh
pip install -U git+https://github.com/GRAAL-Research/deepparse.git@dev
```

------------------

## Cite

Use the following for the article;

```
@misc{yassine2020leveraging,
    title={{Leveraging Subword Embeddings for Multinational Address Parsing}},
    author={Marouane Yassine and David Beauchemin and François Laviolette and Luc Lamontagne},
    year={2020},
    eprint={2006.16152},
    archivePrefix={arXiv}
}
```

and this one for the package;

```
@misc{deepparse,
    author = {Marouane Yassine and David Beauchemin},
    title  = {{Deepparse: A state-of-the-art deep learning multinational addresses parser}},
    year   = {2020},
    note   = {\url{https://deepparse.org}}
}
```

------------------

## Contributing to Deepparse

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a
look at our [contributing guidelines](https://github.com/GRAAL-Research/deepparse/blob/master/CONTRIBUTING.md) for more
details on this matter.

## License

Deepparse is LGPLv3 licensed, as found in
the [LICENSE file](https://github.com/GRAAL-Research/deepparse/blob/master/LICENSE).

------------------

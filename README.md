<div align="center">
<img src="https://raw.githubusercontent.com/GRAAL-Research/deepparse/master/docs/source/_static/logos/deepparse.png" width="220" height="91"/>


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deepparse)](https://pypi.org/project/deepparse)
[![PyPI Status](https://badge.fury.io/py/deepparse.svg)](https://badge.fury.io/py/deepparse)
[![PyPI Status](https://pepy.tech/badge/deepparse)](https://pepy.tech/project/deepparse)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)
[![Continuous Integration](https://github.com/GRAAL-Research/deepparse/workflows/Continuous%20Integration/badge.svg)](https://github.com/GRAAL-Research/deepparse/actions?query=workflow%3A%22Continuous+Integration%22+branch%3Amaster)
[![codecov](https://codecov.io/gh/GRAAL-Research/deepparse/branch/master/graph/badge.svg)](https://codecov.io/gh/GRAAL-Research/deepparse)

[![Download](https://img.shields.io/badge/Download%20Dataset-blue?style=for-the-badge&logo=download)](https://github.com/GRAAL-Research/deepparse-address-data)
  
[![Rate on Openbase](https://badges.openbase.com/python/rating/deepparse.svg)](https://openbase.com/python/deepparse?utm_source=embedded&utm_medium=badge&utm_campaign=rate-badge)
</div>

## Here is deepparse.

Deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepparse to

- Use the pre-trained models to parse multinational addresses with or without attention mechanism,
- retrain our pre-trained models on new data to parse multinational addresses,
- retrain our pre-trained models with your own prediction tags easily,
- retrain a new seq2seq addresses parsing models easily.

Read the documentation at [deepparse.org](https://deepparse.org).

Deepparse is compatible with the __latest version of PyTorch__ and  __Python >= 3.7__.

### Countries and Results

We evaluate our models on two forms of address data

- **clean data** which refers to addresses containing elements from four categories, namely a street name, a
  municipality, a province and a postal code,
- **incomplete data** which is made up of addresses missing at least one category amongst the aforementioned ones.

You can get our dataset [here](https://github.com/GRAAL-Research/deepparse-address-data).

#### Clean Data

The following table presents the accuracy (using clean data) on the 20 countries we used during training for both our
models. Attention mechanisms improve performance by around 0.5% for all countries.

| Country        |   FastText (%) |   BPEmb (%) | Country     |   FastText (%) |   BPEmb (%) |
|:---------------|---------------:|------------:|:------------|---------------:|------------:|
| Norway         |          99.06 |       98.3  | Austria     |          99.21 |       97.82 |
| Italy          |          99.65 |       98.93 | Mexico      |          99.49 |       98.9  |
| United Kingdom |          99.58 |       97.62 | Switzerland |          98.9  |       98.38 |
| Germany        |          99.72 |       99.4  | Denmark     |          99.71 |       99.55 |
| France         |          99.6  |       98.18 | Brazil      |          99.31 |       97.69 |
| Netherlands    |          99.47 |       99.54 | Australia   |          99.68 |       98.44 |
| Poland         |          99.64 |       99.52 | Czechia     |          99.48 |       99.03 |
| United States  |          99.56 |       97.69 | Canada      |          99.76 |       99.03 |
| South Korea    |          99.97 |       99.99 | Russia      |          98.9  |       96.97 |
| Spain          |          99.73 |       99.4  | Finland     |          99.77 |       99.76 |

We have also made a zero-shot evaluation of our models using clean data from 41 other countries; the results are shown
in the next table.

| Country      |   FastText (%) |   BPEmb (%) | Country       |   FastText (%) |   BPEmb (%) |
|:-------------|---------------:|------------:|:--------------|---------------:|------------:|
| Latvia       |          89.29 |       68.31 | Faroe Islands |          71.22 |       64.74 |
| Colombia     |          85.96 |       68.09 | Singapore     |          86.03 |       67.19 |
| Réunion      |          84.3  |       78.65 | Indonesia     |          62.38 |       63.04 |
| Japan        |          36.26 |       34.97 | Portugal      |          93.09 |       72.01 |
| Algeria      |          86.32 |       70.59 | Belgium       |          93.14 |       86.06 |
| Malaysia     |          83.14 |       89.64 | Ukraine       |          93.34 |       89.42 |
| Estonia      |          87.62 |       70.08 | Bangladesh    |          72.28 |       65.63 |
| Slovenia     |          89.01 |       83.96 | Hungary       |          51.52 |       37.87 |
| Bermuda      |          83.19 |       59.16 | Romania       |          90.04 |       82.9  |
| Philippines  |          63.91 |       57.36 | Belarus       |          93.25 |       78.59 |
| Bosnia       |          88.54 |       67.46 | Moldova       |          89.22 |       57.48 |
| Lithuania    |          93.28 |       69.97 | Paraguay      |          96.02 |       87.07 |
| Croatia      |          95.8  |       81.76 | Argentina     |          81.68 |       71.2  |
| Ireland      |          80.16 |       54.44 | Kazakhstan    |          89.04 |       76.13 |
| Greece       |          87.08 |       38.95 | Bulgaria      |          91.16 |       65.76 |
| Serbia       |          92.87 |       76.79 | New Caledonia |          94.45 |       94.46 |
| Sweden       |          73.13 |       86.85 | Venezuela     |          79.23 |       70.88 |
| New Zealand  |          91.25 |       75.57 | Iceland       |          83.7  |       77.09 |
| India        |          70.3  |       63.68 | Uzbekistan    |          85.85 |       70.1  |
| Cyprus       |          89.64 |       89.47 | Slovakia      |          78.34 |       68.96 |
| South Africa |          95.68 |       74.82 |

Moreover, we also tested the performance when using attention mechanism to further improve zero-shot performance on 
those countries; the result are shown in the next table.

| Country       |   FastText (%) |   FastTextAtt (%) |   BPEmb (%) |   BPEmbAtt (%) | Country       |   FastText (%) |   FastTextAtt (%) |   BPEmb (%) |   BPEmbAtt (%) |
|:--------------|---------------:|------------------:|------------:|---------------:|:--------------|---------------:|------------------:|------------:|---------------:|
| Ireland       |          80.16 |             89.11 |       54.44 |          81.84 | Serbia        |          92.87 |             95.88 |       76.79 |           91.4 |
| Uzbekistan    |          85.85 |             87.24 |       70.1  |          76.71 | Ukraine       |          93.34 |             94.58 |       89.42 |          92.65 |
| South Africa  |          95.68 |             97.25 |       74.82 |          97.95 | Paraguay      |          96.02 |             97.08 |       87.07 |          97.36 |
| Greece        |          87.08 |             86.04 |       38.95 |          58.79 | Algeria       |          86.32 |              87.3 |       70.59 |          84.56 |
| Belarus       |          93.25 |             97.4  |       78.59 |          97.49 | Sweden        |          73.13 |             89.24 |       86.85 |          93.53 |
| Portugal      |          93.09 |             94.92 |       72.01 |          93.76 | Hungary       |          51.52 |             51.08 |       37.87 |          24.48 |
| Iceland       |          83.7  |             96.54 |       77.09 |          96.63 | Colombia      |          85.96 |             90.08 |       68.09 |          88.52 |
| Latvia        |          89.29 |             93.14 |       68.31 |          73.79 | Malaysia      |          83.14 |             74.62 |       89.64 |          91.14 |
| Bosnia        |          88.54 |             87.27 |       67.46 |          89.02 | India         |           70.3 |             75.31 |       63.68 |          80.56 |
| Réunion       |          84.3  |             97.74 |       78.65 |          94.27 | Croatia       |           95.8 |             95.32 |       81.76 |          85.99 |
| Estonia       |          87.62 |             88.2  |       70.08 |          77.32 | New Caledonia |          94.45 |             99.61 |       94.46 |          99.77 |
| Japan         |          36.26 |             46.91 |       34.97 |          49.48 | New Zealand   |          91.25 |                97 |       75.57 |           95.7 |
| Singapore     |          86.03 |             89.92 |       67.19 |          88.17 | Romania       |          90.04 |             95.38 |        82.9 |          93.41 |
| Bangladesh    |          72.28 |             78.21 |       65.63 |          77.09 | Slovakia      |          78.34 |             82.29 |       68.96 |             96 |
| Argentina     |          81.68 |             88.59 |       71.2  |          86.8  | Kazakhstan    |          89.04 |             92.37 |       76.13 |          96.08 |
| Venezuela     |          79.23 |             95.47 |       70.88 |          96.38 | Indonesia     |          62.38 |             66.87 |       63.04 |          71.17 |
| Bulgaria      |          91.16 |             91.73 |       65.76 |          93.28 | Cyprus        |          89.64 |             97.44 |       89.47 |          98.01 |
| Bermuda       |          83.19 |             93.25 |       59.16 |          93.8  | Moldova       |          89.22 |             92.07 |       57.48 |          89.08 |
| Slovenia      |          89.01 |             95.08 |       83.96 |          96.73 | Lithuania     |          93.28 |             87.74 |       69.97 |          78.67 |
| Philippines   |          63.91 |             81.94 |       57.36 |          83.42 | Belgium       |          93.14 |             90.72 |       86.06 |          89.85 |
| Faroe Islands |          71.22 |             73.23 |       64.74 |          85.39 |               |                |                   |             |                |

#### Incomplete Data

The following table presents the accuracy on the 20 countries we used during training for both our models but for 
incomplete data. We didn't test on the other 41 countries since we did not train on them and therefore do not expect
to achieve an interesting performance. Attention mechanisms improve performance by around 0.5% for all countries.

| Country        |   FastText (%) |   BPEmb (%) | Country     |   FastText (%) |   BPEmb (%) |
|:---------------|---------------:|------------:|:------------|---------------:|------------:|
| Norway         |          99.52 |       99.75 | Austria     |          99.55 |       98.94 |
| Italy          |          99.16 |       98.88 | Mexico      |          97.24 |       95.93 |
| United Kingdom |          97.85 |       95.2  | Switzerland |          99.2  |       99.47 |
| Germany        |          99.41 |       99.38 | Denmark     |          97.86 |       97.9  |
| France         |          99.51 |       98.49 | Brazil      |          98.96 |       97.12 |
| Netherlands    |          98.74 |       99.46 | Australia   |          99.34 |       98.7  |
| Poland         |          99.43 |       99.41 | Czechia     |          98.78 |       98.88 |
| United States  |          98.49 |       96.5  | Canada      |          98.96 |       96.98 |
| South Korea    |          91.1  |       99.89 | Russia      |          97.18 |       96.01 |
| Spain          |          99.07 |       98.35 | Finland     |          99.04 |       99.52 |

## Getting Started:

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model_type="bpemb", device=0)

# you can parse one address
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

# or multiple addresses
parsed_address = address_parser(
    ["350 rue des Lilas Ouest Québec Québec G1L 1B6", 
    "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

# or multinational addresses
# Canada, US, Germany, UK and South Korea
parsed_address = address_parser(
    ["350 rue des Lilas Ouest Québec Québec G1L 1B6", "777 Brockton Avenue, Abington MA 2351",
     "Ansgarstr. 4, Wallenhorst, 49134", "221 B Baker Street", "서울특별시 종로구 사직로3길 23"])

# you can also get the probability of the predicted tags
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6", 
                                with_prob=True)
```

The predictions tags are the following

- "StreetNumber": for the street number,
- "StreetName": for the name of the street,
- "Unit": for the unit (such as apartment),
- "Municipality": for the municipality,
- "Province": for the province or local region,
- "PostalCode": for the postal code,
- "Orientation": for the street orientation (e.g. west, east),
- "GeneralDelivery": for other delivery information.

### Retrain a Model

> See [here](https://github.com/GRAAL-Research/deepparse/blob/master/examples/fine_tuning.py) for a complete example.

```python
# We will retrain the fasttext version of our pretrained model.
address_parser = AddressParser(model_type="fasttext", device=0)

address_parser.retrain(training_container, 0.8, epochs=5, batch_size=8)

```

### Retrain a Model with an attention mechanism

> See [here](https://github.com/GRAAL-Research/deepparse/blob/master/examples/fine_tuning.py) for a complete example.

```python
# We will retrain the fasttext version of our pretrained model.
address_parser = AddressParser(model_type="fasttext", device=0, attention_mechanism=True)

address_parser.retrain(training_container, 0.8, epochs=5, batch_size=8)

```

### Retrain a Model With New Tags

> See [here](https://github.com/GRAAL-Research/deepparse/blob/master/examples/fine_tuning.py) for a complete example.

```python

address_components = {"ATag":0, "AnotherTag": 1, "EOS": 2}
address_parser.retrain(training_container, 0.8, epochs=1, batch_size=128, 
                       prediction_tags=address_components)

```

### Retrain a Seq2seq model from scratch

> See [here](https://github.com/GRAAL-Research/deepparse/blob/master/examples/retrain_with_new_seq2seq_params.py) for 
> a complete example.

```python

seq2seq_params = {"encoder_hidden_size": 512, "decoder_hidden_size": 512}
address_parser.retrain(training_container, 0.8, epochs=1, batch_size=128, 
                       seq2seq_params=seq2seq_params)

```

### Download our Models

Here are the URLs to download our pre-trained models directly

- [FastText](https://graal.ift.ulaval.ca/public/deepparse/fasttext.ckpt),
- [FastTextAttention](https://graal.ift.ulaval.ca/public/deepparse/fasttext_attention.ckpt),
- [BPEmb](https://graal.ift.ulaval.ca/public/deepparse/bpemb.ckpt),
- [BPEmbAttention](https://graal.ift.ulaval.ca/public/deepparse/bpemb_attention.ckpt),
- [FastText Light](https://graal.ift.ulaval.ca/public/deepparse/fasttext.magnitude.gz) (using [Magnitude Light](https://github.com/davebulaval/magnitude-light)).

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
    title  = {{Deepparse: A State-Of-The-Art Deep Learning Multinational Addresses Parser}},
    year   = {2020},
    note   = {\url{https://deepparse.org}}
}
```

------------------

## Contributing to Deepparse

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a
look at our [contributing guidelines](https://github.com/GRAAL-Research/deepparse/blob/master/.github/CONTRIBUTING.md) for more
details on this matter.

## License

Deepparse is LGPLv3 licensed, as found in
the [LICENSE file](https://github.com/GRAAL-Research/deepparse/blob/master/LICENSE).

------------------

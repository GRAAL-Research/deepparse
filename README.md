<img src="https://raw.githubusercontent.com/GRAAL-Research/deepparse/master/docs/source/_static/logos/logo.png" width="150" height="135"/>

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)
[![Continuous Integration](https://github.com/GRAAL-Research/deepparse/workflows/Continuous%20Integration/badge.svg)](https://github.com/GRAAL-Research/deepparse/actions?query=workflow%3A%22Continuous+Integration%22+branch%3Amaster)

## Here is deepparse.

Deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepparse to:
- Use the pre-trained models to parse multinational addresses.

Read the documentation at [deepparse.org](https://deepparse.org).

Deepparse is compatible with  the __latest version of PyTorch__ and  __Python >= 3.6__.

### Cite
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

## Getting started: 

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model_type="bpemb", device=0)

# you can parse one address
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

# or multiple addresses
parsed_address = address_parser(["350 rue des Lilas Ouest Québec Québec G1L 1B6", "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

# you can also get the probability of the predicted tags
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6", with_prob=True)
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

## Contributing to Deepparse

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a look at our [contributing guidelines](https://github.com/GRAAL-Research/deepparse/blob/master/CONTRIBUTING.md) for more details on this matter.

## License

Deepparse is LGPLv3 licensed, as found in the [LICENSE file](https://github.com/GRAAL-Research/deepparse/blob/master/LICENSE).

------------------

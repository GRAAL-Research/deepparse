![deepParse Logo](https://raw.githubusercontent.com/MAYS3/deepParse/master/docs/source/_static/logos/logo.png)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)
[![Build Status](https://travis-ci.com/MAYAS3/deepParse.svg?token=Zv4ryhyUzUhyJBqsdjui&branch=master)](https://travis-ci.com/MAYAS3/deepParse)

## Here is deepParse.

DeepParse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepParse to:
- Use the pre-trained models to parse multinational addresses.

Read the documentation at [deepParse.org](https://deepparse.org).

DeepParse is compatible with  the __latest version of PyTorch__ and  __Python >= 3.6__.

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
@misc{deepParse,
    author = {Marouane Yassine and David Beauchemin},
    title  = {{DeepParse: A state-of-the-art multinational addresses parser}},
    year   = {2020},
    note   = {\url{https://deepparse.org}}
}
```


------------------

## Getting started: 

```python
from deepparse.parser import AddressParser

address_parser = AddressParser(model="bpemb", device=0)

# you can parse one address
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

# or multiple addresses
parsed_address = address_parser(["350 rue des Lilas Ouest Québec Québec G1L 1B6", "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

# you can also get the probability of the predicted tags
parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6", with_prob=True)
```

------------------

## Installation

Before installing deepParse, you must have the latest version of [PyTorch](https://pytorch.org/) in your environment.

- **Install the stable version of deepParse:**

```sh
pip install deepparse
```

- **Install the latest development version of deepParse:**

```sh
pip install -U git+https://github.com/MAYAS3/deepParse.git@dev
```

## License

deepParse is LGPLv3 licensed, as found in the [LICENSE file](https://github.com/MAYAS3/deepParse/blob/master/LICENSE).

------------------

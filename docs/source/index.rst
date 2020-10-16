.. deepparse documentation master file, created by
   sphinx-quickstart on Sat Feb 17 12:19:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/GRAAL-Research/deepparse

.. meta::
  :description: deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning
  :keywords: deepparse, deep learning, pytorch, neural network, machine learning, natural language processing, parsing, data science, python
  :author: Marouane Yassine & David Beauchemin
  :property="og:image": https://deepparse.org/_static/logos/logo.png

Here is deepparse
=================

Deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepparse to:

- Use the pre-trained models to parse multinational addresses.

Read the documentation at `deepparse.org <https://deepparse.org>`_.

Deepparse is compatible with the **latest version of PyTorch** and  **Python >= 3.6**.

Cite
----
.. code-block:: bib

   @misc{yassine2020leveraging,
       title={{Leveraging Subword Embeddings for Multinational Address Parsing}},
       author={Marouane Yassine and David Beauchemin and François Laviolette and Luc Lamontagne},
       year={2020},
       eprint={2006.16152},
       archivePrefix={arXiv}
   }

and this one for the package;

.. code-block:: bib

   @misc{deepparse,
       author = {Marouane Yassine and David Beauchemin},
       title  = {{deepparse: A state-of-the-art deep learning multinational addresses parser}},
       year   = {2020},
       note   = {\url{https://deepparse.org}}
   }


Getting started
===============
.. code-block:: python

   from deepparse.parser import AddressParser

   address_parser = AddressParser(model="bpemb", device=0)

   # you can parse one address
   parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

   # or multiple addresses
   parsed_address = address_parser(["350 rue des Lilas Ouest Québec Québec G1L 1B6", "350 rue des Lilas Ouest Québec Québec G1L 1B6"])

   # you can also get the probability of the predicted tags
   parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6", with_prob=True)


Installation
============

Before installing deepparse, you must have the latest version of `PyTorch <https://pytorch.org/>`_ in your environment.

- **Install the stable version of deepparse:**

  .. code-block:: sh

   pip install deepparse

- **Install the latest development version of deepparse:**

  .. code-block:: sh

    pip install -U git+https://github.com/GRAAL-Research/deepparse.git@dev


License
=======

deepparse is LGPLv3 licensed, as found in the `LICENSE file <https://github.com/GRAAL-Research/deepparse/blob/master/LICENSE>`_.

API Reference
=============

.. toctree::
  :maxdepth: 1
  :caption: API

  parser

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

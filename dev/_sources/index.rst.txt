:github_url: https://github.com/GRAAL-Research/deepparse


.. meta::

  :description: deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning
  :keywords: deepparse, deep learning, pytorch, neural network, machine learning, natural language processing, parsing, data science, python, address parsing, us address parsing, multilingual address parsing, european address parsing, canadian address parsing, street address parser
  :author: Marouane Yassine & David Beauchemin
  :property="og:image": https://deepparse.org/_static/logos/deepparse.png

.. raw:: html

    <center>
    <a href="https://huggingface.co/datasets/deepparse/worldwide-addresses">
        <img src="https://img.shields.io/badge/Download%20Dataset-blue?style=for-the-badge&logo=download" alt="No message"/></a>
    </center>

Here is Deepparse
=================

Deepparse is a state-of-the-art library for parsing multinational street addresses using deep learning.

Use deepparse to

- parse multinational address using one of our pretrained models with or without attention mechanism,
- parse addresses directly from the command line without code to write,
- parse addresses with our out-of-the-box FastAPI parser,
- retrain our pretrained models on new data to improve parsing on specific country address patterns,
- retrain our pretrained models with new prediction tags easily,
- retrain our pretrained models with or without freezing some layers,
- train a new Seq2Seq addresses parsing models easily using a new model configuration.

Deepparse is compatible with the **latest version of PyTorch** and  **Python >= 3.10, <= 3.14**.

.. only:: not dev_docs

   This documentation covers the latest stable release. The documentation of the development version (``dev``
   branch) is available at `deepparse.org/dev <https://deepparse.org/dev/>`__.

Quick Example
=============

.. code-block:: python

    from deepparse.parser import AddressParser

    address_parser = AddressParser(model_type="bpemb", device=0)

    # you can parse one address
    parsed_address = address_parser("350 rue des Lilas Ouest Québec Québec G1L 1B6")

See :doc:`install/installation` to install Deepparse and :doc:`get_started/get_started` for a complete tour.

Where to Go Next
================

- :doc:`choose_model`: compare the pretrained models (memory, disk, speed, multilingual coverage) to pick one.
- :doc:`results`: the accuracy of our pretrained models on 20 training countries and 41 zero-shot countries,
  with clean and incomplete data.
- :doc:`parser`: the :class:`~deepparse.parser.AddressParser` API, with the memory usage and inference time
  benchmarks.
- :doc:`training_guide` and the examples: retrain a model on your own data.

Cite
====

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
       title  = {{Deepparse: A State-Of-The-Art Deep Learning Multinational Addresses Parser}},
       year   = {2020},
       note   = {\url{https://deepparse.org}}
   }

Contributing to Deepparse
=========================

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a
look at our `contributing guidelines <https://github.com/GRAAL-Research/deepparse/blob/main/.github/CONTRIBUTING.md>`_ for more
details on this matter.

License
=======

Deepparse is LGPLv3 licensed, as found in the `LICENSE file <https://github.com/GRAAL-Research/deepparse/blob/main/LICENSE>`_.

.. toctree::
  :maxdepth: 1
  :caption: Installation

  install/installation


.. toctree::
  :maxdepth: 1
  :caption: Get Started

  get_started/get_started
  choose_model
  results

.. toctree::
  :maxdepth: 1
  :caption: API

  parser
  pre_processor
  dataset_container
  comparer
  embeddings_models
  download_tools
  errors
  cli
  api

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Examples

  examples/parse_addresses
  examples/parse_addresses_uri
  examples/parse_addresses_with_cli
  examples/retrained_model_parsing
  examples/fine_tuning
  examples/fine_tuning_uri
  examples/fine_tuning_with_csv_dataset
  examples/retrain_attention_model
  examples/retrain_with_new_prediction_tags
  examples/retrain_with_new_seq2seq_params
  examples/single_country_retrain

.. toctree::
  :maxdepth: 1
  :caption: Model training

  training_guide

.. toctree::
  :maxdepth: 1
  :caption: Project

  changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`

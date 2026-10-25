.. role:: hidden
    :class: hidden-section

.. _download_tools:

Models Download
===============

.. currentmodule:: deepparse.download_tools

The :class:`~deepparse.parser.AddressParser` downloads the pretrained weights and the embeddings it needs the first time
it is used (in ``~/.cache/deepparse`` by default). The following functions download them ahead of time, for example
before sending a model to a node without Internet access (then use ``offline=True`` in the
:class:`~deepparse.parser.AddressParser`). The ``download_model`` command of the :doc:`cli` does the same from the
command line. They can be imported directly from ``deepparse`` (e.g. ``from deepparse import download_model``).

Pretrained Models
-----------------

.. autofunction:: download_model

.. autofunction:: download_models

.. autofunction:: download_weights

Embeddings
----------

.. autofunction:: download_fasttext_embeddings

.. autofunction:: download_fasttext_magnitude_embeddings

.. autofunction:: load_fasttext_embeddings

Sample Data
-----------

A small sample of tagged addresses of one country can be downloaded from the
`worldwide-addresses dataset <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ and used with a
:class:`~deepparse.dataset_container.ListDatasetContainer` (see :ref:`dataset_container`).

.. autofunction:: download_worldwide_addresses_sample

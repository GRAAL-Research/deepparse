.. role:: hidden
    :class: hidden-section

Embeddings Models
=================

.. currentmodule:: deepparse.embeddings_models

The embeddings models convert each word of an address into a vector that the Seq2Seq network then tags. The
:class:`~deepparse.parser.AddressParser` creates the right embeddings model for its ``model_type``, so you only need
these classes to use the embeddings directly or to build your own pipeline:

- the FastText models (``"fasttext"`` and ``"fasttext-attention"``) use :class:`FastTextEmbeddingsModel`, which reads
  the French `fastText <https://fasttext.cc/>`__ ``.bin`` embeddings,
- the FastText Light models (``"fasttext-light"``) use :class:`MagnitudeEmbeddingsModel`, which reads the same
  embeddings in the `Magnitude <https://github.com/plasticityai/magnitude>`__ format,
- the BPEmb models (``"bpemb"`` and ``"bpemb-attention"``) use :class:`BPEmbEmbeddingsModel`, which uses the
  `byte-pair multilingual subword <https://nlp.h-its.org/bpemb/>`__ embeddings.

The FastText ``.bin`` embeddings are read by :class:`FastTextBinReader`, our own reader of the fastText binary format.
It gives the same vectors as the ``fasttext`` package without depending on ``fasttext`` nor ``gensim``, and its input
matrix is memory-mapped: it loads in a few seconds, only the rows used are kept in RAM, and the data loading workers
share its memory. It replaces the ``fasttext``/``gensim`` loader used up to version 0.11.0.

Interface
---------

.. autoclass:: EmbeddingsModel
    :members:

Implementations
---------------

.. autoclass:: FastTextEmbeddingsModel
    :members:

    .. automethod:: __call__

.. autoclass:: MagnitudeEmbeddingsModel
    :members:

    .. automethod:: __call__

.. autoclass:: BPEmbEmbeddingsModel
    :members:

    .. automethod:: __call__

FastText Binary Reader
----------------------

.. autoclass:: FastTextBinReader
    :members: get_dimension, get_word_id, get_subword_ids, get_word_vector

Factory
-------

.. autoclass:: EmbeddingsModelFactory
    :members:

.. role:: hidden
    :class: hidden-section

Parser
======

.. currentmodule:: deepparse.parser

Pre-trained complete model
--------------------------

This is the complete pre-trained address parser model. This model allows using the pre-trained weights to predict the
tags of any address.

We offer, for now, only two pre-trained models, FastText and BPEmb. The first one relies on
`fastText <https://fasttext.cc/>`_ French pre-trained embeddings to parse the address and the second use
the `byte-pair multilingual subword <https://nlp.h-its.org/bpemb/>`_ pre-trained embeddings. In both cases,
the architecture is similar, and performances are comparable; our results are available in this
`article <https://arxiv.org/abs/2006.16152>`_. But note that we have cherry-picked the best trained model for each,
so on the contrary of our article results, BPemb gives the best results than fastText. the following table present the
memory usage and time performance of both models.

.. list-table::
        :header-rows: 1

        *   - Layer Type
            - Output size
            - # of Parameters
        *   - Input
            - 1x28x28
            - 0


.. autoclass:: AddressParser
    :members:

    .. automethod:: __call__


.. autoclass:: ParsedAddress
    :members:

.. role:: hidden
    :class: hidden-section

Parser
======

.. currentmodule:: deepparse.parser

Pre-trained Complete Model
--------------------------

This is the complete pre-trained address parser model. This model allows using the pre-trained weights to predict the
tags of any address.

We offer, for now, only two pre-trained models, FastText and BPEmb. The first one relies on
`fastText <https://fasttext.cc/>`_ French pre-trained embeddings to parse the address and the second use
the `byte-pair multilingual subword <https://nlp.h-its.org/bpemb/>`_ pre-trained embeddings. In both cases,
the architecture is similar, and performances are comparable; our results are available in this
`article <https://arxiv.org/abs/2006.16152>`_.

Memory Usage and Time Performance
*********************************

We have conducted an experiment, and we report the results in the next two tables. In each table, we report the RAM usage,
and in the first table, we also report the GPU memory usage. Also, for both table, we report the mean-time of execution
that was obtained by processing ~183,000 address using different batch size (2^0, ..., 2^9)
(i.e. :math:`\frac{\text{Total time to process all addresses}}{~183,000} =` time per address).
In addition, we proposed a lighter version (fastText-light) of our fastText model using
`Magnitude embeddings mapping <https://github.com/plasticityai/magnitude>`_. Fot this lighter model, in average results
are a little bit lower on trained country (around ~2%) but are similar on zero-shot country
(see our `article <https://arxiv.org/abs/2006.16152>`_ for more details).

.. list-table::
        :header-rows: 1

        *   -
            - Memory usage GPU (GB)
            - Memory usage RAM (GB)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText [1]_
            - ~1
            - ~8
            - ~0.00236
            - ~0.0004
        *   - fastTextAttention
            - ~1.1
            - ~8
            - ~0.0052
            - ~0.0010
        *   - fastText-light
            - ~1
            - ~1
            - ~0.0028
            - ~0.0037
        *   - BPEmb
            - ~1
            - ~1
            - ~0.0053
            - ~0.0019
        *   - BPEmbAttention
            - ~1.1
            - ~1
            - ~0.0086
            - ~0.0027
        *   - Libpostal
            - N/A
            - N/A
            - <1
            - ~0.00007

.. [1] Note that on Windows, we use the Gensim Fasttext models that use ~10 GO with similar performance.


.. list-table::
        :header-rows: 1

        *   -
            - Memory usage RAM (GB)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText [2]_
            - ~8
            - ~0.0168
            - ~0.0026
        *   - fastTextAttention
            - ~8
            - ~0.0212
            - ~0.0054
        *   - fastText-light
            - ~1
            - ~0.0170
            - ~0.0030
        *   - BPEmb
            - ~1
            - ~0.0219
            - ~0.0059
        *   - BPEmbAttention
            - ~1
            - ~0.0270
            - ~0.0075
        *   - Libpostal
            - N/A
            - <1
            - ~0.00007

.. [2] Note that on Windows, we use the Gensim Fasttext models that use ~10 GO with similar performance.

The two tables highlight that the batch size (number of address in the list to be parsed) influence the processing time.
Thus, the more there is address, the faster processing each address can be. You can also improve performance by
using more worker for the data loader created with your data within the call. But note that this performance
improvements is not linear.

AddressParser
-------------

.. autoclass:: AddressParser
    :members:

    .. automethod:: __call__

Formatted Parsed Address
------------------------

.. autoclass:: FormattedParsedAddress
    :members:

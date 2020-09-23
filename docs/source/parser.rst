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
so on the contrary of our article results, BPemb gives the best results than fastText.

Memory Usage and Time Performance
*********************************

We have conducted an experiment, and we report the results in the next two tables. In each table, we report the RAM usage,
and in table I, we also report the GPU's memory usage. Also, for both table, we report the mean-time of execution
that was obtained by processing ~183,000 address using different batch size (2^0, ..., 2^9)
(i.e. :math:`\frac{\text{Total time to process all addresses}}{~183,000} = \text{time per address}`).

.. list-table::
        :header-rows: 1

        *   -
            - Memory usage GPU (GO)
            - Memory usage RAM (GO)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText
            - ~0.885
            - ~9
            - ~0.0037
            - ~0.0007
        *   - BPEmb
            - ~0.885
            - ~2
            - ~0.0097
            - ~0.0045

.. list-table::
        :header-rows: 1

        *   -
            - Memory usage RAM (GO)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText
            - ~9
            - ~0.0037
            - ~0.0007
        *   - BPEmb
            - ~2
            - ~0.0309
            - ~0.0075

The two tables highlight that the batch size (number of address in the list to be parsed) influence the processing time.
Thus, the more there is address, the faster processing each address can be. However, note that at some point, this
'improvement' of performance decrease. We found that fastText and BPEmb obtain their best performance using a batch
size of 256, beyond that performance decrease. But, these results were not rigorously tested. For example, using
the fastText model, our test shown that parsing a single address (batch of 1 element) takes around 0.003 seconds.
This time can be reduced to 0.00033 seconds per address when using a batch of 256, but using 512 take 0.0035 seconds.

.. autoclass:: AddressParser
    :members:

    .. automethod:: __call__


.. autoclass:: ParsedAddress
    :members:

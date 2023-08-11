.. role:: hidden
    :class: hidden-section

Parser
======

.. currentmodule:: deepparse.parser

Pre-trained Complete Model
--------------------------

This is the complete pretrained address parser model. This model allows using the pretrained weights to predict the
tags of any address.

We offer, for now, only two pretrained models, FastText and BPEmb. The first one relies on
`fastText <https://fasttext.cc/>`_ French pretrained embeddings to parse the address, and the second use
the `byte-pair multilingual subword <https://nlp.h-its.org/bpemb/>`_ pretrained embeddings. In both cases,
the architecture is similar, and performances are comparable; our results are available in this
`article <https://arxiv.org/abs/2006.16152>`_.

Memory Usage and Time Performance
*********************************

To assess memory usage and inference time performance, we have conducted an experiment using Linux OS, Python 3.11,
Torch 2.0 and CUDA 11.7 (done March 21, 2023). The next two tables report the results. In each table,
we report the RAM usage, and in the first table, we also report the GPU memory usage.
Also, for both tables, we report the mean-time of execution
that was obtained by processing ~183,000 addresses using different batch sizes (2^0, ..., 2^9)
(i.e. :math:`\frac{\text{Total time to process all addresses}}{~183,000} =` time per address).
In addition, we proposed a lighter version (fasttext-light) of our fastText model using
`Magnitude embeddings mapping <https://github.com/plasticityai/magnitude>`_. For this lighter model, on average, results
are a little bit lower for the trained country (around ~2%) but are similar for the zero-shot country
(see our `article <https://arxiv.org/abs/2006.16152>`_ for more details).

.. list-table::
        :header-rows: 1

        *   - With a GPU
            - Memory usage GPU (GB)
            - Memory usage RAM (GB)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText [1]_
            - ~1
            - ~8
            - ~0.0023
            - ~0.0004
        *   - fastTextAttention
            - ~1.1
            - ~8
            - ~0.0043
            - ~0.0007
        *   - fastText-light
            - ~1
            - ~1
            - ~0.0028
            - ~0.0037
        *   - BPEmb
            - ~1
            - ~1
            - ~0.0055
            - ~0.0015
        *   - BPEmbAttention
            - ~1.1
            - ~1
            - ~0.0081
            - ~0.0019
        *   - Libpostal
            - N/A
            - N/A
            - N/A
            - ~0.00004

.. [1] Note that on Windows, we use the Gensim FastText models with ~10 GO with similar performance.


.. list-table::
        :header-rows: 1

        *   - With a CPU
            - Memory usage RAM (GB)
            - Mean time of execution (batch of 1) (s)
            - Mean time of execution (batch of more than 1) (s)
        *   - fastText [2]_
            - ~8
            - ~0.0128
            - ~0.0026
        *   - fastTextAttention
            - ~8
            - ~0.0230
            - ~0.0057
        *   - fastText-light
            - ~1
            - ~0.0170
            - ~0.0030
        *   - BPEmb
            - ~1
            - ~0.0179
            - ~0.0044
        *   - BPEmbAttention
            - ~1
            - ~0.0286
            - ~0.0075
        *   - Libpostal
            - N/A
            - N/A
            - ~0.00004

.. [2] Note that on Windows, we use the Gensim FastText models that use ~10 GO with similar performance.

Thus, the more address is, the faster each address can be processed. You can also improve performance by using more
workers for the data loader created with your data within the call. But note that this performance improvement is not linear.
Furthermore, as of version ``0.9.6``, we now use Torch 2.0 and many other tricks to improve
processing performance. Here a few: if the parser uses a GPU, it will pin the memory in the Dataloader and reduce some
operations (e.g. useless ``.to(device)``).

AddressParser
-------------

.. autoclass:: AddressParser
    :members:

    .. automethod:: __call__

Formatted Parsed Address
------------------------

.. autoclass:: FormattedParsedAddress
    :members:

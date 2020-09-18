.. role:: hidden
    :class: hidden-section

Network
=======

.. currentmodule:: deepparse.network

Pre-trained Seq2Seq model interface
-----------------------------------

.. autoclass:: PreTrainedSeq2SeqModel
    :members:

Pre-trained Seq2Seq model
-------------------------

.. autoclass:: PreTrainedFastTextSeq2SeqModel
    :members:

    .. automethod:: __call__

.. autoclass:: PreTrainedBPEmbSeq2SeqModel
    :members:

    .. automethod:: __call__

Seq2Seq components
------------------

.. autoclass:: Encoder
    :members:

.. autoclass:: Decoder
    :members:

.. autoclass:: EmbeddingNetwork
    :members:

.. _choose_model:

Choose a Model
==============

Deepparse offers pretrained models built on two embeddings, FastText and BPEmb, each with or without an attention
mechanism, plus a lighter FastText version. You select one with the ``model_type`` (and ``attention_mechanism``)
argument of :class:`~deepparse.parser.AddressParser`. All the figures below come from this documentation; their
source is given under each table.

Model Types
-----------

.. list-table::
        :header-rows: 1
        :widths: 22 18 60

        *   - ``model_type``
            - Alias
            - Description
        *   - ``"fasttext"``
            - ``"fastest"``
            - Seq2Seq model using the French `fastText <https://fasttext.cc/>`__ pretrained embeddings. The
              ``"fastest"`` alias is the quickest model to process one address.
        *   - ``"fasttext-attention"``
            -
            - The ``"fasttext"`` model with an attention mechanism. Requires ``attention_mechanism=True`` (same
              model as ``model_type="fasttext", attention_mechanism=True``).
        *   - ``"fasttext-light"``
            - ``"lightest"``
            - The FastText model using the `Magnitude <https://github.com/plasticityai/magnitude>`__ embeddings
              mapping, which reduces the memory footprint. The ``"lightest"`` alias is the model using the least RAM
              and GPU memory.
        *   - ``"bpemb"``
            - ``"best"``
            - Seq2Seq model using the `byte-pair multilingual subword <https://nlp.h-its.org/bpemb/>`__ pretrained
              embeddings. The ``"best"`` alias (the default ``model_type``) is described as the best accuracy
              performance.
        *   - ``"bpemb-attention"``
            -
            - The ``"bpemb"`` model with an attention mechanism. Requires ``attention_mechanism=True`` (same model as
              ``model_type="bpemb", attention_mechanism=True``).

Source: the ``model_type`` argument of :class:`~deepparse.parser.AddressParser` and the model names accepted by the
parser (case-insensitive).

Resources and Speed
-------------------

.. list-table::
        :header-rows: 1

        *   - Model
            - Embeddings on disk
            - RAM (GB)
            - GPU memory (GB)
            - Time per address, CPU (batch of 1 / more than 1) (s)
            - Time per address, GPU (batch of 1 / more than 1) (s)
        *   - fastText
            - 6.8 GO
            - ~1.3 [#memory_mapped]_
            - ~1
            - ~0.0128 / ~0.0026
            - ~0.0023 / ~0.0004
        *   - fastTextAttention
            - 6.8 GO
            - ~1.3 [#memory_mapped]_
            - ~1.1
            - ~0.0230 / ~0.0057
            - ~0.0043 / ~0.0007
        *   - fastText-light
            - 3.3 GO
            - ~1
            - ~1
            - ~0.0170 / ~0.0030
            - ~0.0028 / ~0.0037
        *   - BPEmb [#bpemb_gpu]_
            - 116 MB
            - ~1
            - ~1
            - ~0.0179 / ~0.0044
            - ~0.0055 / ~0.0015
        *   - BPEmbAttention [#bpemb_gpu]_
            - 116 MB
            - ~1
            - ~1.1
            - ~0.0286 / ~0.0075
            - ~0.0081 / ~0.0019

Sources:

- Embeddings on disk: the note of :class:`~deepparse.parser.AddressParser` (the FastText embeddings take 6.8 GO, the
  FastText-light embeddings 3.3 GO and BPEmb 116 MB). The pretrained weights take at most 44 MB in addition.
- RAM, GPU memory and time per address: the "Memory Usage and Time Performance" tables of the :doc:`parser` page
  (Linux, Python 3.11, Torch 2.0 and CUDA 11.7, ~183,000 addresses, March 21, 2023). The attention mechanism takes
  about 100 MB more on GPU.

.. [#memory_mapped] Measured in October 2026 (Linux, Python 3.12, Torch 2.14, CPU) with the memory-mapped
   :class:`~deepparse.embeddings_models.FastTextBinReader`: ~1.3 GB at peak for the whole process while parsing 10,000
   addresses, on every OS (see :doc:`embeddings_models`). With the former ``gensim`` loader (up to version 0.11.0), the
   same parser took ~8.4 GB (13.2 GB at peak during the loading).

.. [#bpemb_gpu] On GPU, the BPEmb models parse batches of addresses about twice as fast after version 0.11.0,
   since their embedding network runs the sub-word LSTM on all the words of a batch in one call. Measured in October
   2026 (RTX 6000 Ada, Torch 2.6, batches of 256): 3,000 addresses are parsed in ~1.0-1.3 s instead of ~2.4-2.6 s,
   with or without attention, with the same predictions. The times of the table are from the March 2023 experiment.
   On CPU, the time is unchanged.


Accuracy and Multilingual Coverage
----------------------------------

All the models were trained on addresses from the same 20 countries; the :doc:`results` page gives their accuracy on
those countries and a zero-shot evaluation on 41 other countries.

- **FastText** uses French word embeddings with character n-grams. On the 20 training countries (clean data), it
  reaches 98.9% to 99.97% accuracy (see :doc:`results`).
- **BPEmb** uses multilingual subword embeddings, which cover the vocabulary of many languages. On the 20 training
  countries (clean data), it reaches 96.97% to 99.99% accuracy (see :doc:`results`).
- **Attention mechanism**: it improves the accuracy by around 0.5% on the training countries, and it further improves
  the zero-shot accuracy on most of the 41 other countries (see the attention table of :doc:`results`).
- **FastText Light**: on average, its results are about 2% lower on the training countries, but similar on the
  zero-shot countries (see the :doc:`parser` page and our `article <https://arxiv.org/abs/2006.16152>`__).

When to Use Each Model
----------------------

- **You parse addresses from the 20 training countries and want the quickest model**: ``"fasttext"`` (``"fastest"``).
  Its embeddings take the most disk space (6.8 GO).
- **You parse addresses from other countries (zero-shot)**: prefer a model with an attention mechanism
  (``"fasttext-attention"`` or ``"bpemb-attention"``) and check the country in the zero-shot tables of
  :doc:`results`, since the accuracy varies a lot between countries.
- **Disk space is limited** (e.g. a container image or a small node): ``"bpemb"`` or ``"bpemb-attention"``, whose
  embeddings take 116 MB.
- **You want the default model**: ``"bpemb"`` (``"best"``).
- **You want the FastText model with a lower memory footprint**: ``"fasttext-light"`` (``"lightest"``). Note that a
  FastText Light model cannot be retrained nor tested, and cannot use data loading workers (``num_workers`` > 0) on
  Windows (see :class:`~deepparse.errors.FastTextModelError`).
- **You need to retrain a model** on new data or new tags: any model except ``"fasttext-light"`` (see the
  :doc:`training_guide`).

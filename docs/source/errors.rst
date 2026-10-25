.. role:: hidden
    :class: hidden-section

Errors
======

.. currentmodule:: deepparse.errors

Deepparse raises the following exceptions, in addition to the standard Python ones (e.g. ``ValueError``).

DataError
---------

Raised by the :ref:`dataset containers <dataset_container>` and the parser when the addresses or the tags are not
valid (e.g. ``None``, empty or whitespace-only addresses).

.. autoclass:: DataError

FastTextModelError
------------------

Raised by :class:`~deepparse.parser.AddressParser` when an operation is not supported with a FastText Light
(``"fasttext-light"``) model, namely retraining or testing such a model, or using data loading workers (``num_workers`` > 0) with it on Windows.

.. autoclass:: FastTextModelError

ServerError
-----------

.. autoclass:: ServerError

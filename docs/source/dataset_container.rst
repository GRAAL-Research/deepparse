.. role:: hidden
    :class: hidden-section

Dataset Container
=================

.. currentmodule:: deepparse.dataset_container

Interface
*********

.. autoclass:: DatasetContainer
    :members:

Implementations
***************
.. autoclass:: PickleDatasetContainer
    :members:

.. autoclass:: CSVDatasetContainer
    :members:

.. autoclass:: ListDatasetContainer
    :members:

Dataset Validation Steps
************************

We also applied data validations to all data containers using the following three functions.

.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_empty
.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_whitespace_only
.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_none

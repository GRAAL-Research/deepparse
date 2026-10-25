.. _dataset_container:

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

Sample Data
***********

A small sample of tagged addresses of one country can be downloaded from the
`worldwide-addresses dataset <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ and used with a
:class:`~deepparse.dataset_container.ListDatasetContainer`.

.. autofunction:: deepparse.download_tools.download_worldwide_addresses_sample

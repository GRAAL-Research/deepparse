.. role:: hidden
    :class: hidden-section

Dataset Container
=================

.. currentmodule:: deepparse.dataset_container

To load the data, we use dataset containers implemented with respect to an interface.

.. autoclass:: DatasetContainer
    :members:

.. autoclass:: PickleDatasetContainer
    :members:

.. autoclass:: CSVDatasetContainer
    :members:

We also applied data validations to our data containers using the following three functions.

.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_empty
.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_whitespace_only
.. autofunction:: deepparse.data_validation.data_validation.validate_if_any_none

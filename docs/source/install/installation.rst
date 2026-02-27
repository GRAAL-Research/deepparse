.. role:: hidden
    :class: hidden-section

Installation
============

Deepparse is available for Python 3.10 and Python 3.11.

.. note::
   We do not recommend installation as a root user on your system Python.
   Please setup a virtual environment, *e.g.*, via `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.

Quick Start
-----------

.. raw:: html
   :file: quick-start.html

Installation
------------

Before installing deepparse, you must have the latest version of `PyTorch <https://pytorch.org/>`_ in your environment.

- **Install the stable version of Deepparse:**

  .. code-block:: sh

   pip install deepparse

- **Install the stable version of Deepparse with the app extra dependencies:**

  .. code-block:: sh

   pip install "deepparse[app]"

- **Install the stable version of Deepparse with all extra dependencies:**

  .. code-block:: sh

   pip install "deepparse[all]"

- **Install the latest development version of Deepparse:**

  .. code-block:: sh

    pip install -U git+https://github.com/GRAAL-Research/deepparse.git@dev
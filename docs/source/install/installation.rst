.. role:: hidden
    :class: hidden-section

Installation
============

Deepparse is available for Python 3.10, Python 3.11, Python 3.12, Python 3.13 and Python 3.14.

.. note::
   We do not recommend installation as a root user on your system Python.
   Please setup a virtual environment, *e.g.*, via `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_, or create a `Docker image <https://www.docker.com/>`_.

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

Versioning
----------

Deepparse follows `semantic versioning <https://semver.org/>`_: given a version ``MAJOR.MINOR.PATCH``, a patch
release only fixes bugs, a minor release adds features in a backward compatible way, and only a major release can
remove or change a public API in a backward incompatible way.

The public API is made of the documented classes and functions and, for the ``deepparse`` top-level package, of the
names listed in ``deepparse.__all__``. Deprecated APIs emit a ``DeprecationWarning`` and are kept at least until the
next major version.

The following APIs are deprecated and will be removed in 1.0.0:

- ``deepparse.BPEmbBaseURLWrapperBugFix`` and the ``deepparse.bpemb_url_bug_fix`` module (deprecated in 0.12.0): use
  ``deepparse.BPEmbLoader`` (``deepparse.bpemb_loader`` module), which has the same arguments and behaviour.
- ``deepparse.load_fasttext_embeddings`` (deprecated in 0.12.0): use
  ``deepparse.embeddings_models.FastTextBinReader``, which it already returns.
- ``deepparse.FASTTEXT_AVAILABLE`` (deprecated in 0.12.0): Deepparse no longer needs the ``fasttext`` package to load
  the FastText embeddings, so this flag is no longer used.
- The internal helpers and constants that are reachable as ``deepparse.<name>`` but are not in
  ``deepparse.__all__`` (e.g. ``deepparse.weights_init`` or ``deepparse.MODEL_REPO_IDS``). They are not part of the
  public API: import them from their own module instead.

Since 0.12.0, the standard library and third-party modules that the wildcard imports of ``deepparse`` re-exported by
accident (e.g. ``deepparse.os``, ``deepparse.torch`` or ``deepparse.np``) are no longer reachable from ``deepparse``,
and ``from deepparse import *`` only imports the names of ``deepparse.__all__``.
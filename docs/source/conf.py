# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from deepparse import __version__ as version

year = str(datetime.datetime.now().year)

# -- Project information -----------------------------------------------------

project = "deepparse"
copyright = "2020-" + year + ", Marouane Yassine & David Beauchemin"
author = "Marouane Yassine & David Beauchemin"

# The short X.Y version
version = version
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Third-party targets that have no Sphinx inventory, ignored by the nitpicky mode (-n).
nitpick_ignore = [
    ("py:class", "S3Path"),
    ("py:class", "cloudpathlib.s3.s3path.S3Path"),
]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"deepparse {release}"

# The logo is yellow on a transparent background, so it reads well in the light and dark modes.
html_logo = "_static/logos/deepparse.png"
html_favicon = "_static/logos/favicon.png"

html_theme_options = {
    # The logo already contains the project name.
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/GRAAL-Research/deepparse/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

html_static_path = ["_static"]

html_extra_path = ["CNAME", "favicon.ico"]

# -- Intersphinx mappings ----------------------------------------------------

intersphinx_mapping = {
    "PyTorch": ("https://docs.pytorch.org/docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Extension configuration -------------------------------------------------

autodoc_default_options = {"member-order": "bysource"}

# sphinx-copybutton: do not copy the shell and Python REPL prompts.
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

# myst-parser: generate anchors for the Markdown headings (e.g. the changelog versions).
myst_heading_anchors = 2

# -- Linkcheck ---------------------------------------------------------------

linkcheck_timeout = 30
linkcheck_retries = 2
# Anchors rendered by JavaScript that linkcheck cannot see.
linkcheck_anchors_ignore_for_url = [
    r"https://docs\.pytorch\.org/.*",
    r"https://poutyne\.org/.*",
    r"https://github\.com/.*/blob/.*",
]

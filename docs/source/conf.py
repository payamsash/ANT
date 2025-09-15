# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))
project = 'ANT'
copyright = '2025, Payam S. Shabestari'
author = 'Payam S. Shabestari'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # pull in docstrings
    "sphinx.ext.napoleon",      # Google/NumPy style docstrings
    "sphinx.ext.intersphinx",   # link to external docs (Python, NumPy…)
    "sphinx_autodoc_typehints", # nice rendering of type hints
    "numpydoc",
]
# extensions.append("sphinx_gallery.gen_gallery")

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
html_static_path = ['_static']

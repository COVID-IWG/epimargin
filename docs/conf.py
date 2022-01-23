# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

try:
    # version > 3.0
    from sphinx.ext.apidoc import main as apidoc_main
except ImportError:
    # sphinx version < 3
    from sphinx.apidoc import main as apidoc_main


# -- Project information -----------------------------------------------------

project = "epimargin"
copyright = "2021, Satej Soman, Caitlin Loftus, Steven Buschbach, Manasi Phadnis, Luís M. A. Bettencourt"
author = "Satej Soman, Caitlin Loftus, Steven Buschbach, Manasi Phadnis, Luís M. A. Bettencourt"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

autoclass_content = "both"

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show_inheritance": True,
    "inherited-members": True,
}

# hack to generate apidoc automatically.
cwd = Path(__file__).parent
module = cwd / ".." / "epimargin"
os.environ["SPHINX_APIDOC_OPTIONS"] = ",".join(
    [f"{k}={v}" for k, v in autodoc_default_options.items()]
)
apidoc_main(["-e", "-o", str(cwd), str(module), "--force"])

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

import os
import re
import sys
import importlib
from docutils import nodes
from docutils.parsers.rst import Directive
from pathlib import Path

# Minimum version, enforced by sphinx
needs_sphinx = "4.3"


# This is a nasty hack to use platform-agnostic names for types in the
# documentation.

# must be kept alive to hold the patched names
_name_cache = {}


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

sys.path.insert(0, "../../src")

# sys.path.insert(0, os.path.abspath("../../src/nastro"))
# print(sys.path)
# sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.ifconfig",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"
master_doc = "index"

# General substitutions.
project = "nastro"
copyright = "2024, Alfonso Sánchez Rodríguez"
author = "Alfonso Sánchez Rodríguez"
release = "0.0.1"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%B %d, %Y"

# List of documents that shouldn't be included in the build.
# unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# Ensure all our internal links work
nitpicky = True
nitpick_ignore = [
    # This ignores errors for classes (OptimizeResults, sparse.dok_matrix)
    # which inherit methods from `dict`. missing references to builtins get
    # ignored by default (see https://github.com/sphinx-doc/sphinx/pull/7254),
    # but that fix doesn't work for inherited methods.
    ("py:class", "a shallow copy of D"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "(k, v), remove and return some (key, value) pair as a"),
    ("py:class", "None.  Update D from dict/iterable E and F."),
    ("py:class", "v, remove specified key and return the corresponding value."),
]

exclude_patterns = []  # glob-style

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

# html_favicon = "_static/favicon/favicon.ico"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo_small.png",
        "image_dark": "_static/logo_small.png",
        "text": "Nastro!",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/alfonsoSR/nastro",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nastro",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_align": "content",
    "show_nav_level": 1,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "use_edit_page_button": True,
    "show_version_warning_banner": True,
    "back_to_top_button": True,
}

html_sidebars = {"index": [], "**": ["sidebar-nav-bs"]}

html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_css_files = ["custom.css"]
html_js_files = ["custom-icon.js"]
html_context = {"default_mode": "light"}
html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = ".html"

htmlhelp_basename = "nastro"

# # Autodoc
# autodoc_default_options = {
#     "inherited-members": True,
#     "member-order": "bysource",
#     "show-inheritance": False,
# }
# autodoc_typehints = "description"
# autodoc_type_aliases = {
#     "Double": "nastro.types.core.Double",
#     "Vector": "nastro.types.core.Vector",
#     "ArrayLike": "nastro.types.core.ArrayLike",
# }

autodoc_default_options = {
    "inherited-members": None,
    "member-order": "bysource",
}
autodoc_typehints = "none"


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------
autosummary_generate = True

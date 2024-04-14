import os
import re
import sys
import importlib
from docutils import nodes
from docutils.parsers.rst import Directive

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

sys.path.insert(0, os.path.abspath("../sphinxext"))

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
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

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

# html_favicon = "_static/favicon/favicon.ico"

html_theme_options = {
    "logo": {"text": "Nastro: Numerical Astrodynamics"},
    "github_url": "https://github.com/alfonsoSR/nastro",
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
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_nav_level": 1,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "use_edit_page_button": False,
    "show_version_warning_banner": True,
    "back_to_top_button": True,
}

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

if "sphinx.ext.pngmath" in extensions:
    pngmath_use_preview = True
    pngmath_dvipng_args = ["-gamma", "1.5", "-D", "96", "-bg", "Transparent"]

mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"

plot_html_show_formats = False
plot_html_show_source_link = False

# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# XeLaTeX for better support of unicode characters
latex_engine = "xelatex"

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

latex_elements = {}

# Additional stuff for the LaTeX preamble.
latex_elements[
    "preamble"
] = r"""
\newfontfamily\FontForChinese{FandolSong-Regular}[Extension=.otf]
\catcode`琴\active\protected\def琴{{\FontForChinese\string琴}}
\catcode`春\active\protected\def春{{\FontForChinese\string春}}
\catcode`鈴\active\protected\def鈴{{\FontForChinese\string鈴}}
\catcode`猫\active\protected\def猫{{\FontForChinese\string猫}}
\catcode`傅\active\protected\def傅{{\FontForChinese\string傅}}
\catcode`立\active\protected\def立{{\FontForChinese\string立}}
\catcode`业\active\protected\def业{{\FontForChinese\string业}}
\catcode`（\active\protected\def（{{\FontForChinese\string（}}
\catcode`）\active\protected\def）{{\FontForChinese\string）}}

% In the parameters section, place a newline after the Parameters
% header.  This is default with Sphinx 5.0.0+, so no need for
% the old hack then.
% Unfortunately sphinx.sty 5.0.0 did not bump its version date
% so we check rather sphinxpackagefootnote.sty (which exists
% since Sphinx 4.0.0).
\makeatletter
\@ifpackagelater{sphinxpackagefootnote}{2022/02/12}
    {}% Sphinx >= 5.0.0, nothing to do
    {%
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
% but expdlist old LaTeX package requires fixes:
% 1) remove extra space
\usepackage{etoolbox}
\patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
% 2) fix bug in expdlist's way of breaking the line after long item label
\def\breaklabel{%
    \def\@breaklabel{%
        \leavevmode\par
        % now a hack because Sphinx inserts \leavevmode after term node
        \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
    }%
}
    }% Sphinx < 5.0.0 (and assumed >= 4.0.0)
\makeatother

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

% Fix footer/header
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
"""

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False

# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = "dump.xml"

# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

# Autodoc
autodoc_default_options = {
    "inherited-members": True,
    "member-order": "bysource",
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_type_aliases = {
    "Double": "nastro.types.core.Double",
    "Vector": "nastro.types.core.Vector",
    "ArrayLike": "nastro.types.core.ArrayLike",
}

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plot_pre_code = """
import numpy as np
np.random.seed(0)
"""
plot_include_source = True
plot_formats = [("png", 100), "pdf"]

import math

phi = (math.sqrt(5) + 1) / 2

plot_rcparams = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

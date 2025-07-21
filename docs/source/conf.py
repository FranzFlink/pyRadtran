# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyradtran'
copyright = '2025, pyradtran developers'
author = 'pyradtran developers'

# Get version from package
try:
    from pyradtran import __version__
    version = __version__
    release = __version__
except ImportError:
    version = '0.1.0'
    release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',  # Adds support for GitHub Pages
    # 'sphinx_autoapi.extension',  # Temporarily disabled
    'nbsphinx',
    'myst_parser',
]

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

# Temporarily disabled autoapi settings
# autoapi_type = 'python'
# autoapi_dirs = ['../../pyradtran']
# autoapi_keep_files = True

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Include notebooks directory in source path
import os
import shutil
def setup(app):
    """Setup function to copy notebooks to source directory"""
    notebooks_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'notebooks')
    notebooks_dest = os.path.join(os.path.dirname(__file__), 'notebooks')
    
    if os.path.exists(notebooks_src):
        if os.path.exists(notebooks_dest):
            shutil.rmtree(notebooks_dest)
        shutil.copytree(notebooks_src, notebooks_dest, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.ipynb_checkpoints', 'work', 'data'))

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom sidebar
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# Additional options for GitHub Pages
html_baseurl = 'https://franzflink.github.io/pyRadtran/'
html_title = f"{project} v{release} documentation"

# -- Options for Napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for nbsphinx extension ------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Allow notebooks with errors to build
nbsphinx_timeout = 300  # Timeout for notebook execution in seconds

# Configure notebook display settings
nbsphinx_kernel_name = 'python3'
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# Custom CSS for notebooks
nbsphinx_prolog = r"""
{% set docname = 'notebooks/' + env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::
        This page was generated from `{{ docname|e }} <https://github.com/FranzFlink/pyRadtran/blob/main/{{ docname|e }}>`_.
        Interactive online version: :raw-html:`<a href="https://mybinder.org/v2/gh/FranzFlink/pyRadtran/main?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# Notebook epilog
nbsphinx_epilog = r"""
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ env.doc2path(env.docname, base=None) | escape_latex }}}} ends here.}}
"""

# -- Options for MyST parser ---------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "linkify",
    "strikethrough",
    "substitution",
    "tasklist"
]

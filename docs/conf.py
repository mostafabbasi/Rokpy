
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from rokpy import __version__

project = 'rokpy'
copyright = '2025, Mostafa Abbasi'
author = 'Mostafa Abbasi'
version = __version__
release = __version__
myst_heading_anchors = 3  # Auto-generate heading anchors up to level 3

html_title = f"{project} Documentation"  # Affects browser tab title
html_short_title = "Docs" 


html_theme = 'furo'
# html_theme = 'sphinx_material'
# html_theme = 'cloud'
# html_theme = 'sphinx_rtd_theme'

html_favicon = '_static/rokpy_logo.png'
html_logo = "_static/rokpy.png" 

html_static_path = ["_static"]
# html_css_files = ["furo.css"]

html_theme_options = {
    # 'navigation_depth': 1,
    # "show_navbar_depth": 3,
    # "max_navbar_depth": 3,
    # 'collapsiblesidebar ': True,
}



python_display_short_literal_types = False
maximum_signature_line_length = 90
wrap_signatures_with_parens = True
suppress_warnings = ["toc.not_readable", "docutils","ref.footnote","toc.not_readable","toc.not_included"]
# add_function_parentheses = False
toc_object_entries_show_parents = 'hide'
# html_split_index = True
html_compact_lists = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True

latex_use_latexmk = False
latex_engine = 'xelatex'


extensions = [
    'sphinx.ext.autodoc',        # Auto-generate from docstrings
    'sphinx.ext.napoleon',       # Support Google/NumPy docstring styles
    'sphinx.ext.mathjax',        # Math rendering
    'sphinx.ext.intersphinx',    # Cross-reference Python docs
    'sphinx.ext.autosummary',    # Generate summary tables
    'myst_parser',
    "nbsphinx",
        # 'sphinx.ext.viewcode',       # Add source code links
]
nbsphinx_execute = 'never'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "colon_fence",      # Use ::: for code fences
    "deflist",          # Definition lists
    "dollarmath",       # LaTeX math
    "html_image",       # HTML img tags
    "replacements",     # Text replacements
    # "linkify",          # Auto-detect links
]


autodoc_default_options = {
    "members": True,
    "no-inherited-members": True,
}
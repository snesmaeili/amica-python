"""Sphinx configuration for the jamica documentation."""

from __future__ import annotations

import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_SOURCE_DIR.parent

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "jamica"
author = "jamica developers"

_today = date.today()
copyright = f"2024-{_today.year}, {author}. Last updated on {_today.isoformat()}"

try:
    version = get_version("jamica")
except PackageNotFoundError:
    version = "0.1.0"

release = version


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
]

nitpicky = False
keep_warnings = True


# ---------------------------------------------------------------------------
# Autodoc / Autosummary / Numpydoc
# ---------------------------------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_attributes_as_param_list = True
numpydoc_class_members_toctree = False


# ---------------------------------------------------------------------------
# MyST Markdown
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3


# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


# ---------------------------------------------------------------------------
# Copy button
# ---------------------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False


# ---------------------------------------------------------------------------
# Sphinx Gallery
# ---------------------------------------------------------------------------

sphinx_gallery_conf = {
    "doc_module": "jamica",
    "reference_url": {
        "jamica": None,
    },
    "examples_dirs": str(REPO_ROOT / "examples"),
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "generated",
    "filename_pattern": r"^plot_|^[0-9]+_",
    "ignore_pattern": r"validation/.*|cluster/.*",
    "run_stale_examples": False,
    "remove_config_comments": True,
    "within_subsection_order": "FileNameSortKey",
}


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "jamica"
html_short_title = "jamica"
html_show_sphinx = False
html_show_copyright = True

html_static_path = ["_static"]

html_theme_options = {
    # The navbar scales its logo to roughly 40px, which is far too short for
    # the stacked wordmark to survive, so the navbar gets the emblem alone and
    # the theme prints the project name beside it. The emblem is almost
    # entirely chromatic and would nearly survive a single file for both
    # themes -- except the pianist's stool is black and would drop out of the
    # dark navbar, hence the pair.
    "logo": {
        "image_light": "_static/logo-mark.png",
        "image_dark": "_static/logo-mark-dark.png",
        "alt_text": "jamica",
    },
    "github_url": "https://github.com/snesmaeili/jamica",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 6,
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/snesmaeili/jamica",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/jamica/",
            "icon": "fa-brands fa-python",
        },
    ],
}

html_context = {
    "github_user": "snesmaeili",
    "github_repo": "jamica",
    "github_version": "main",
    "doc_path": "docs",
}

# The navbar logo is set through html_theme_options["logo"] above rather than
# html_logo, because that is the only form pydata-sphinx-theme reads for its
# light/dark pair. All of these are generated by scripts/make_logo_assets.py.
html_favicon = "_static/favicon.ico"

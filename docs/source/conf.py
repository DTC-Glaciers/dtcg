# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# pylint: skip-file
# flake8: noqa: F541

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

# -- Remove Copyright Boilerplate --------------------------------------------


def remove_boilerplate(app, what, name, obj, options, lines):
    if what == "module":
        del lines[:16]


def setup(app):
    app.connect("autodoc-process-docstring", remove_boilerplate)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DTCG"
copyright = f"{date.today().year}, DTCG Contributors"
author = "DTCG Contributors"
release = "0.5.0"
version = os.environ.get("READTHEDOCS_VERSION", "latest")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # "shibuya",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autodoc configuration ---------------------------------------------------

# autodoc_member_order = 'bysource'
autodoc_default_flags = [
    "members",
    "undoc-members",
    "show-inheritance",
]
autodoc_typehints = "description"
autodoc_type_aliases = {
    "GlacierDirectory": "oggm.GlacierDirectory",
    "GeoZarrHandler": "dtcg.datacube.geozarr.GeoZarrHandler",
}
autosummary_generate = True

numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

napoleon_use_param = True
napoleon_use_rtype = True

pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "breeze"
html_static_path = ["_static"]

# support_icon = """<img src="https://github.com/DTC-Glaciers/dtc-glaciers.github.io/blob/main/img/favicon.png"></img>"""
html_theme_options = {
    "sidebar_secondary": False,
    # "external_links": [
    #     {
    #         "name": "DTC Glaciers",
    #         "url": "https://dtcglaciers.org",
    #         "html": support_icon,
    #     },
    # ],
}
html_context = {
    "github_user": "DTC-Glaciers",
    "github_repo": "dtcg",
    "github_version": "main",
    "doc_path": "docs",
    "current_version": version,
}

master_doc = "index"
trim_footnote_reference_space = True
html_show_sphinx = False
html_static_path = ["_static"]
highlight_language = "python"
html_title = "DTCG"
html_logo = "./_static/dtcg_logo_transparent.png"
html_favicon = "./_static/favicon.ico"

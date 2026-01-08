.. _installation:

===============
Getting started
===============

.. _requirements:

Requirements
============

DTCG is compatible with Python 3.11 and above on Linux, MacOS, and WSL.

.. warning::
    The DTCG API is still experimental.
    Features may change and break at any time.

.. _install_instructions:

Installation
============

The recommended environment managers are ``conda``/``mamba``, or ``uv``.

DTCG requires a working OGGM installation in your active Python environment.
If you do not have this already set up, pass the ``oggm`` flag when installing.

Activate your virtual environment, and install as an editable:

.. code-block:: bash

    git clone git@github.com:DTC-Glaciers/dtcg.git
    cd dtcg
    pip install -e .[oggm]  # if you do not have OGGM installed
    uv pip install -e .[oggm]  # if using uv


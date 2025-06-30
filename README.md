# dtcg
Python API for DTC-Glaciers access and analysis.

[![Tests](https://github.com/DTC-Glaciers/dtcg/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/DTC-Glaciers/dtcg/actions/workflows/run-tests.yml)

## Installation

Supports Python 3.11 and 3.12.

If you already installed OGGM's dependencies:

```
pip install -e .
```
This includes a DTCG-compatible fork of OGGM.
You will need to reinstall OGGM if you would like to use your own fork.

If you do not have a working OGGM environment:
```
pip install -e .[oggm]
```
This will install all of OGGM's dependencies.

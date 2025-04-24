# dtcg
Python API for DTC-Glaciers access and analysis

## Installation

```
pip install -e .
```
This also installs a DTCG-compatible fork of OGGM.
You will need to reinstall OGGM if you would like to use your own fork.

If you do not have a working OGGM environment:
```
pip install -e .[oggm]
```

This will install all of OGGM's dependencies, but not OGGM itself.

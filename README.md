# ML Reproducibility Challenge 2023

Code for the submission to the ML Reproducibility Challenge 2023.

# Getting Started

We use Python version 3.10 for this repository.

We use [Poetry](https://python-poetry.org/) for dependency management. More specifically version `1.2.0`.

After installing Poetry, run the following command to create a virtual environment and install
all dependencies:

```shell
poetry install
```

You can then activate the virtual environment using:

```shell
poetry shell
```

# Experiments

We use [DVC](https://dvc.org/) to run the experiments and track their results.

To reproduce all results use:

```shell
dvc exp run
```

The parameters and configuration values can be found inside the [conf](conf/)
directory.

`params.yaml` is derived from those files and should not be modified manually.

## Data Valuation

```shell
dvc exp run data-valuation
```

# Contributing

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```


## Fetching datasets

To fetch a dataset simply call `csshapley22/data/fetch.py`, which uses the `datasets` section
of `params.yaml` to download and caches the corresponding data files. The string-based kwargs 
configuration is used as an identifier, to verify later that a configuration has
changed.
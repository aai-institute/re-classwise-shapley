# Reproduction "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification"

Code for the submission to the ML Reproducibility Challenge 2023. The original paper can
be found [here](https://arxiv.org/abs/2106.06860). 

# Getting Started

We use Pytho 3.10 and to install the library install
all dependencies:

```shell
pip install
```

# Reproduction

## Run all experiments

```
dvc repro
```

## Run selected experiments

```shell
dvc exp run \
  --set-param active.experiments=[point_removal,noise_removal] \
  --set-param active.datasets=[cifar10,click,covertype,cpu,diabetes,fmnist_binary,mnist_binary,mnist_multi,phoneme] \
  --set-param active.valuation_methods=[loo,classwise_shapley,beta_shapley,tmc_shapley,banzhaf_shapley,owen_sampling_shapley,least_core] \
  --set-param active.models=[logistic_regression,knn,svm,gradient_boosting_classifier] \
  --set-param active.repetitions=[1,2,3,4,5]
```

# Contributing

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```

## Fetching datasets

To fetch a dataset simply call `re_classwise_shapley/data/fetch.py`, which uses
the `datasets` section
of `params.yaml` to download and caches the corresponding data files. The string-based
kwargs
configuration is used as an identifier, to verify later that a configuration has
changed.
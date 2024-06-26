# Reproduction "CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification"

Code for the submission to TMLR 2024. The original paper can be found [here](https://arxiv.org/abs/2211.06800).

# Getting started

## Installation

By using

```shell
conda create --name re-classwise-shapley python=3.10
conda activate re-classwise-shapley
pip install poetry
poetry install
```

a new conda environment is created and all dependencies get installed. 


## env file

In `sample.env` is an example for a env. This environment has to be copied to `.env` 
file prior to starting the

### MLflow Configuration

#### Tracking Server URL
- `MLFLOW_TRACKING_URI`: The URL where the MLflow tracking server is hosted.
  - **Example**: `http://localhost:5000`

#### Artifact Storage
- `MLFLOW_S3_ENDPOINT_URL`: The endpoint URL for S3 compatible artifact storage.
  - **Example**: `http://localhost:9000`

### AWS Credentials (Used for MinIO in this setup)
- `AWS_ACCESS_KEY_ID`: The access key ID for AWS or AWS-compatible services.
  - **Example**: `key`
- `AWS_SECRET_ACCESS_KEY`: The secret access key for AWS or AWS-compatible services.
  - **Example**: `access_key`

### MinIO Configuration
- `MINIO_ROOT_USER`: The username for the MinIO root user.
  - **Example**: `mlflow`
- `MINIO_ROOT_PASSWORD`: The password for the MinIO root user.
  - **Example**: `password`

### MySQL Database Configuration

#### mlflow Database
- `MYSQL_DATABASE`: The database name for MLflow.
  - **Example**: `mlflow`
- `MYSQL_USER`: The username for accessing the MLflow database.
  - **Example**: `mlflow`
- `MYSQL_PASSWORD`: The password for accessing the MLflow database.
  - **Example**: `password`

#### MySQL Root User
- `MYSQL_ROOT_PASSWORD`: The root password for the MySQL database.
  - **Example**: `password`

### Security Note
Ensure that these credentials are secured and managed appropriately to prevent unauthorized access.


## Start mlflow using docker

For experiment tracking we use MLflow. To start MLflow 

```shell
cp .env docker/mlflow/.env
cd docker/mlflow
docker compose up -d
```

and open `http://localhost:5000` in your browser. MLflow relies on a S3 bucket served by
a minio server. All plots and artifacts are logged to this bucket. If you want to stop 
MLflow execute

```shell
docker compose down
```

## Run experiments

Execute

```
dvc init
dvc exp run
```

to run all experiments. For more details on how to run the experiments have a look at
the [Reproduction](#Reproduction) section.

# Pipeline

## Stages

The pipeline is defined in the `dvc.yaml` and consists of the following six stages:

| Stages              | Description                                                                      |
|---------------------|----------------------------------------------------------------------------------|
| 1. Fetch data       | Fetches a dataset with a specific ID from openml.                                |
| 2. Preprocess data  | Applies filters and preprocessors to each dataset as defined in `params.yaml`    |
| 3. Sample data      | Use a seed to perform stratified sampling on the preprocessed data.              |
| 4. Calculate values | Compute values for a sampled dataset from (1).                                   |
| 5. Evaluate curves  | Calculates several curves based on the values calculated in (4)                  |
| 6. Evaluate metrics | Calculates several metrics based on the curves calculated in (5)                 |
| 7. Render plots     | Renders plots, saves to disk and logs to MLflow. Used information from (1) to (5) |

Each stage requires inputs and outputs their result to a sub-folder of `output` folder. 
In the following section we describe each stage in more detail.

### 1. Fetch data 

The first stage fetches a dataset from openml. The dataset is identified by an ID. For
technical reasons, a dataset might be fetched twice, if it is preprocessed in two
different ways. This happens for `mnist_binary` and `mnist_multi`. Each dataset is 
stored inside of `output/raw/<dataset_name>`. In each of those folders is a numpy array
`x.npy` (containing the features) and a numpy array `y.npy` (containing the targets). 
Furthermore, there is a `info.json` file containing some meta information about the 
dataset, like feature names, target names and a description of the dataset.

### 2. Preprocess data

Takes data from the previous stage and applies filters and preprocessors to it. The 
filters and preprocessors are defined in `params.yaml`. The preprocessed data is stored
inside of `output/preprocessed/<dataset_name>`. In each of those folders is a numpy 
array `x.npy` (containing the features) and a numpy array `y.npy` (containing the 
targets). Furthermore, there is a `info.json` file containing some meta information 
about the dataset and the distribution of the labels. At this point we can be sure that 
the type of `y.npy` is `int` and the dataset represents a classification problem. 
Furthermore, files `preprocess.json` and `filter.json` are stored and contain information
about which pre-processors and filters were used.

### 3. Sample data

Each repetition id is treated as the initial entropy for the random generator. This 
stage takes the dataset from the previous stage, sub-samples a dataset as defined in 
the `sampler` section of the experiment. Before storing it on disk, the dataset applies
a sample preprocessor, e.g. flip labels and stores side information as well. The outputs
are stored in `output/sampled/<experiment_name>/<dataset_name>/<repetition_id>` and 
contains a `val_set.pkl`, `test_set.pkl` and an (optional) `preprocess_info.json`. Both
datasets contain the same training samples but differ in their validation samples. The 
`json` file might contain information about the preprocessor, e.g. the indices of labels
flipped.

### 4. Calculate values

This stage takes the sampled data and calculates values for each valuation method. The 
valuation results are stored in 
`output/values/<experiment_name>/<model_name>/<dataset_name>/<repetition_id>/`. Each
applied method generates two files. The first file has the name
`valuation.<method_name>.pkl` and contains the valuation results. The second file has
the name `valuation.<method_name>.stats.json` and contains meta information, e.g. the
execution time. Again the repetition id is used an initial seed.

### 5. Evaluate curves

After the values are calculated, the curves need to be evaluated. In general there
can be multiple metrics for one curve. The metrics are defined in the 
`curves` section of the experiment. Per metric a file is generated in 
`output/curves/<experiment_name>/<model_name>/<dataset_name>/<repetition_id>/
<valuation_method_name>`. The first file contains the aggregated result (a single 
number) with the file name `<metric_name>.csv`.

### 6. Evaluate metrics

After the curves are calculated, the metrics need to be evaluated. In general there
can be multiple metrics for one curve. The metrics are defined in the 
`metrics` section of the experiment. Per metric a file is generated in 
`output/metrics/<experiment_name>/<model_name>/<dataset_name>/<repetition_id>/
<valuation_method_name>`. The first file contains the aggregated result (a single 
number) with the file name `<metric_name>.<curve_name>.csv`.

### 7. Render plots

Last but not least, the plots are rendered and all relevant information is logged to
MLflow. The following plots are generated

| Plot             | Description                                                                              |
|------------------|------------------------------------------------------------------------------------------|
| Histogram        | Histogram for each method and each dataset in comparison to TMCS.                        |
| Time (Boxplot)   | Running times per method and dataset in the style of a boxplot.                          |
| Metric           | A table per metric (in the style of a heatmap) containing the mean over all repetitions. |
| Metric (Boxplot) | A boxplot per metric with the mean and variance for each valuation method compared.      |
| Curves           | A plot per metric comparing the curves per method on each dataset                        |

## Reproduction

In general there are two ways of running the experiments. The original way uses `dvc` to 
execute the pipeline. However, writing and reading the `dvc.lock` file takes some time. 
Hence, an alternative way runs experiments from a python script directly. Both ways can
be bridged by using `dvc commit`. 

### Manual runs

Sometimes `dvc` takes a lot of time inbetween stages. Hence, we integrated an option to
run the experiments without `dvc` and committing the results later on. Execute

```shell
python scripts/run_pipeline.py
dvc commit
```

to do a manual and faster run without `dvc`. Committing the results is optional, but
necessary if you want to switch back to `dvc` with the results of the run. 


### Run with `dvc`

Data version control (dvc) is used to manage the experiments. It internally caches 
results in `.dvc` folder and uses the `dvc.lock` file to track the dependencies of the
stages. For more information on `dvc` please consult their 
[documentation](https://dvc.org/doc).

#### Run all experiments

To run all experiments. Execute the following command:

```shell
dvc exp run
```

#### Run a subset of experiments

Inside the `params.yaml` file there is a section called `active`. This section can be
used to select a subset of the models, valuation methods, datasets, experiments and
seeds. They can be either modified directly in `params.yaml` or by using

```shell
dvc exp run \
  --set-param active.experiments=[point_removal,noise_removal] \
  --set-param active.datasets=[cifar10,click,covertype,cpu,diabetes,fmnist_binary,mnist_binary,mnist_multi,phoneme] \
  --set-param active.valuation_methods=[loo,classwise_shapley,beta_shapley,tmc_shapley,banzhaf_shapley,owen_sampling_shapley,least_core] \
  --set-param active.models=[logistic_regression,knn,svm,gradient_boosting_classifier] \
  --set-param active.repetitions=[1,2,3,4,5]
```

the `set-param` flag when calling `dvc exp run`

#### Run using `dvc repro`

Alternatively you can use to execute a certain stage only. In practice, validation
checks are skipped and thus the command runs faster than `dvc exp run`.

```shell
dvc repro [-s <stage>]
```

### Manual

Sometimes `dvc` takes a lot of time inbetween stages. Hence, we integrated an option to
run the experiments without `dvc` and committing the results later on. Execute

```shell
python scripts/run_pipeline.py
dvc commit
```

to do a manual and faster run without `dvc`. Committing the results is optimal, but is
necessary if you want to switch back to `dvc` with the results of the run. 

# Development

Make sure to install the pre-commit hooks:

```shell
pre-commit install
```

## Extending the benchmark

While the aim of this repository is to reproduce the aforementioned paper, you can
modify the benchmark to your needs. In the following section we describe how to add
different parts and include them in the experiments.


### Add a new dataset

Adding a dataset is very ease. It only requires to add a new entry to the `datasets`
section:

```yaml
datasets:
  mnist_binary:
    openml_id: 554
    filters:
      binarization:
        label_zero: '1'
        label_one: '7'
    preprocessor:
      principal_resnet_components:
        n_components: 32
        grayscale: true
        seed: 103
```

The `openml_id` is used to fetch the dataset from openml. The `filters` section is used
to filter the dataset based on labels. In the example above, the dataset is filtered to
only contain labels `1` and `7`. The `preprocessor` section is used to describe how
to preprocess the dataset. In the example above, the dataset is preprocessed using a 
principal component analysis (PCA) on extracted resnet18 features. The `seed` is used to
make the results repeatable.


### Add a new model

To add a new model (with preprocessing), modify the function 
`re_classwise_shapley.model.instantiate_model`. For example

```python
[...]
elif model == "logistic_regression":
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(**model_kwargs, random_state=random_state),
    )
[...]
```

defines a logistic regression model with a `StandardScaler`. After specifying a new
model in `instantiate_model`, add the model to the `models` section of `params.yaml`:

```yaml
models:
  logistic_regression:
    model: logistic_regression
    solver: liblinear

  logistic_regression_balanced:
    model: logistic_regression
    solver: liblinear
    class_weight: balanced
```

Here the model is used two times with different keyword arguments. One has the name
`logistic_regression` and the other `logistic_regression_balanced`. The name is used to
identify the model throughout the pipeline. The keyword arguments are passed to the
model inside of `instantiate_model`.

### Add a new valuation method 

To add a new valuation method, modify the function
`re_classwise_shapley.valuation_methods.compute_values`:

```python
match valuation_method:
    [...]
    case "loo":
        return compute_loo(utility, n_jobs=n_jobs, progress=progress)
    [...]
```

Afterward define your model in `params.yaml`:

```yaml
valuation_methods:
  loo:
    algorithm: loo
    progress: true
```

### Register a new filter, preprocessor or sample preprocessor.

If you need new filters they can be registered in the dictionary called 
`FilterRegistry` residing in file `re_classwise_shapley.filters`. The same holds
for preprocessors. They can be registered in the dictionary called `PreprocessorRegistry`
residing in file `re_classwise_shapley.preprocess`. After a dataset is sampled, sample
preprocessors get applied. They can be registered in the dictionary called 
`SamplePreprocessorRegistry` residing in file `re_classwise_shapley.preprocess`, e.g.
flip some labels. 

### Add a new experiment

An experiment can be defined as:

```yaml
experiments:
  noise_removal:
    sampler: default
    preprocessors:
       [...]
    metrics:  
       [...]
```

Note that each experiment has a unique name (in our case `noise_removal`). Furthermore,
one has to define a sampling strategy. In our case we use the `default` sampler and it
is defined in the `params.yaml` as well:

```yaml
samplers:
  default:
    train: 0.1667
    val: 0.1667
    test: 0.6667
    max_samples: 3000
```

### Register a new curve

To add a new metric, register your curve with `CurveRegistry` in
`re_classwise_shapley.curve`. See existing
curves for more details. After a curve is registered it can be used in the `params.yaml`
file as follows:

```yaml
experiments:
  point_removal:
    [...]
    curves:
      accuracy:
        fn: accuracy
        type: "mse"
```

A special role has the parameter. It defines how much of the curve
should be drawn in the plots. It is not passed to the metric itself, but used in the 
last stage. All other parameters are passed as keyword arguments to the metric.

### Register a new metric

To add a new metric, register your metric with `MetricRegistry` in
`re_classwise_shapley.metric`. A metric can accept any subset of parameters 
`data`, `values`, `info`, `n_jobs`, `config`, `progress` and `seed`. See existing
metrics for more details. After a metric is registered it can be used in the `params.yaml`
file as follows:

```yaml
experiments:
  point_removal:
    [...]
    metrics:
      mean:
        curve:
          - accuracy
        fn: mean_accuracy
        arg: arg
```

The parameter `len_curve_perc` has a special role. It defines how much of the curve
should be drawn in the plots. It is not passed to the metric itself, but used in the 
last stage. All other parameters are passed as keyword arguments to the metric.


# The reproducibility report

The report is written using [TeXmacs](https://www.texmacs.org). The sources
can be found in the `report` folder.

Due to the requirement to use TMLR's style files, we use TeXmacs' *conservative
export* to LaTeX. After activating this preference in the editor, import the
LaTeX file and work on it as if it were a regular TeXmacs file. Save the file to
export back, while preserving any TeXmacs-specific formatting. Export to PDF
from TeXmacs any time, or use the LaTeX file with TMLR's template project for
overleaf.

## Images

Images are linked from the `output` folder. These are not included in the
repository, but can be generated from saved run data using

```shell
dvc repro render-plots -f -s
```

For convenience, the script under `scripts/finish_latex_export.py` can be used
to gather all used images for easy upload to overleaf. It will create a folder
`report/output` mimicking the structure of the `output` folder, and copy all
used images there.

## Caveats

TMLR requires using \citep and \citet for citations. TeXmacs does not support
this natively, so we use an ugly workaround. In TeXmacs we enclose the required 
citations in parentheses, and then we use a simple regex to convert those. This
is just a hack until we add a couple of macros to texmacs for proper export.

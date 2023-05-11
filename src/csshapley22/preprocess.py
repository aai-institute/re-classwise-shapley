import pickle
from functools import partial
from typing import Dict

from csshapley22.data.config import Config
from csshapley22.data.utils import make_hash_sha256
from csshapley22.log import setup_logger
from csshapley22.types import (
    ModelGeneratorFactory,
    ValTestSetFactory,
    ValuationMethodsFactory,
)
from csshapley22.utils import instantiate_model
from csshapley22.valuation_methods import compute_values

logger = setup_logger()


def parse_valuation_methods_config(
    valuation_methods: Dict[str, Dict]
) -> ValuationMethodsFactory:
    logger.info("Parsing valuation methods...")
    return {
        name: partial(
            compute_values,
            valuation_method=name,
            **kwargs,
        )
        if kwargs is not None
        else partial(compute_values, valuation_method=name)
        for name, kwargs in valuation_methods.items()
    }


def parse_datasets_config(dataset_settings: Dict[str, Dict]) -> ValTestSetFactory:
    logger.info("Parsing datasets...")
    collected_datasets = {}
    for dataset_name, dataset_kwargs in dataset_settings.items():
        validation_set, test_set = load_single_dataset(dataset_name, dataset_kwargs)
        collected_datasets[dataset_name] = (validation_set, test_set)

    return collected_datasets


def load_single_dataset(dataset_name: str, dataset_kwargs: Dict):
    dataset_idx = make_hash_sha256(dataset_kwargs)
    raw_folder = Config.PREPROCESSED_PATH / dataset_idx
    validation_set_path = str(raw_folder / "validation_set.pkl")
    test_set_path = str(raw_folder / "test_set.pkl")

    with open(validation_set_path, "rb") as file:
        validation_set = pickle.load(file)

    with open(test_set_path, "rb") as file:
        test_set = pickle.load(file)

    return validation_set, test_set


def parse_models_config(models_config: Dict[str, Dict]) -> ModelGeneratorFactory:
    logger.info("Parsing models...")
    collected_generators = {}
    for model_name, model_kwargs in models_config.items():
        collected_generators[model_name] = partial(
            instantiate_model, model_name=model_name, **model_kwargs
        )

    return collected_generators

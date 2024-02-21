import click
from dvc.api import params_show

from csshapley22.constants import RANDOM_SEED
from csshapley22.data.fetch import fetch_dataset
from csshapley22.log import setup_logger
from csshapley22.utils import set_random_seed

logger = setup_logger()
set_random_seed(RANDOM_SEED)


@click.command()
def fetch_data():
    logger.info("Starting downloading of data.")

    params = params_show()
    general_settings = params["general"]

    # fetch datasets
    datasets_settings = general_settings["datasets"]
    for dataset_name, dataset_kwargs in datasets_settings.items():
        logger.info(f"Fetching dataset {dataset_name} with kwargs {dataset_kwargs}.")
        fetch_dataset(dataset_name, dataset_kwargs)


if __name__ == "__main__":
    fetch_data()

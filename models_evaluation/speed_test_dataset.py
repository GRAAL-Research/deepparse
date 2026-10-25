"""
Tools to build the speed test dataset used by the speed test evaluation scripts.

The original speed test dataset (a pickle of ~183,000 tagged addresses) was hosted on a server that is no longer
available. We now build an equivalent dataset from the `worldwide-addresses
<https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ HuggingFace dataset, using the
:func:`~deepparse.download_worldwide_addresses_sample` function.

The dataset is built from a fixed list of countries (the 20 countries our models were trained on) with the same number
of addresses per country, in a fixed order. ``download_worldwide_addresses_sample`` is deterministic (it always reads
the end of the last parquet file of a country), so two runs with the same number of addresses (and the same dataset
revision) process exactly the same addresses. No shuffling is done; thus, no random seed is needed.
"""

import os
import pickle
from typing import List, Tuple

from deepparse import download_worldwide_addresses_sample

# The 20 countries our models were trained on (see models_evaluation/tools.py). In the original data, the "kp.p" file
# holds South Korean addresses, which are under the "kr" country code in the worldwide-addresses dataset.
SPEED_TEST_COUNTRIES = [
    "br",
    "us",
    "kr",
    "ru",
    "de",
    "fr",
    "nl",
    "ch",
    "fi",
    "es",
    "cz",
    "gb",
    "mx",
    "no",
    "ca",
    "it",
    "au",
    "dk",
    "pl",
    "at",
]

# The number of addresses processed in the original speed test (see docs/source/parser.rst).
DEFAULT_NUM_ADDRESSES = 183000

DEFAULT_DATA_DIRECTORY = "./data"


def build_speed_test_dataset(
    num_addresses: int = DEFAULT_NUM_ADDRESSES, data_directory: str = DEFAULT_DATA_DIRECTORY
) -> List[Tuple[str, List]]:
    """
    Build (or load from the cache) the speed test dataset.

    The addresses are evenly split between the ``SPEED_TEST_COUNTRIES`` (the first countries get one more address when
    ``num_addresses`` is not a multiple of the number of countries). The dataset is cached as a pickle in
    ``data_directory`` so that the following runs do not download it again.

    Args:
        num_addresses (int): The number of addresses of the dataset. If a country holds fewer addresses than its
            share, the dataset will contain fewer addresses. The default value is ``183000``.
        data_directory (str): The directory where to cache the dataset. The default value is ``"./data"``.

    Return:
        A list of tuples ``(address, tags)``.
    """
    if num_addresses <= 0:
        raise ValueError(f"The number of addresses must be a positive integer, got {num_addresses}.")

    cache_path = os.path.join(data_directory, f"speed_test_dataset_{num_addresses}.p")
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    num_addresses_per_country, remainder = divmod(num_addresses, len(SPEED_TEST_COUNTRIES))
    dataset = []
    for idx, country_code in enumerate(SPEED_TEST_COUNTRIES):
        num_samples = num_addresses_per_country + (1 if idx < remainder else 0)
        if num_samples == 0:
            continue
        country_samples = download_worldwide_addresses_sample(country_code=country_code, num_samples=num_samples)
        if len(country_samples) < num_samples:
            print(f"Only {len(country_samples)} addresses available for {country_code} ({num_samples} requested).")
        dataset.extend(country_samples)

    os.makedirs(data_directory, exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump(dataset, file)

    return dataset

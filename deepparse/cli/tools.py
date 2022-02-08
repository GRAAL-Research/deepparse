from typing import List

import pandas as pd


def is_csv_path(path: str) -> bool:
    return ".csv" in path


def is_pickle_path(dataset_path: str) -> bool:
    return ".p" in dataset_path or ".pickle" in dataset_path


def to_csv(parsed_address: List[str], export_path: str, sep: str) -> None:
    pd.DataFrame(parsed_address).to_csv(export_path, sep=sep, index=False)

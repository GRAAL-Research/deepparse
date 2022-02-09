# pylint: disable=too-many-arguments, self-assigning-variable

import pickle
from typing import List

import pandas as pd
import torch

from deepparse.dataset_container import DatasetContainer


def create_file(path: str, content: str):
    # pylint: disable=unspecified-encoding
    """ "
    Function to create a file for test
    """
    with open(path, "w") as f:
        f.write(content)


BATCH_SIZE = 32


class ADataContainer(DatasetContainer):
    def __init__(self, is_training_container: bool = True):
        super().__init__(is_training_container=is_training_container)
        self.data = (torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))


address_len = 6
base_string = "an address with the number {}"
a_tags_sequence = ["tag1", "tag2", "tag2", "tag3", "tag3", "tag4"]
default_csv_column_name = ["Address", "Tags"]


def create_data(number_of_data_points: int = 4, predict_container: bool = False) -> List:
    if predict_container:
        file_content = [base_string.format(str(data_point)) for data_point in range(number_of_data_points)]
    else:
        file_content = [
            (base_string.format(str(data_point)), a_tags_sequence) for data_point in range(number_of_data_points)
        ]

    return file_content


def create_pickle_file(path: str, number_of_data_points: int = 4, predict_container: bool = False) -> None:
    pickle_file_content = create_data(number_of_data_points, predict_container)

    with open(path, "wb") as f:
        pickle.dump(pickle_file_content, f)


def create_csv_file(
    path: str,
    predict_container: bool = False,
    number_of_data_points: int = 4,
    column_names=None,
    separator="\t",
    reformat_list_fn=None,
) -> None:
    csv_file_content = create_data(number_of_data_points, predict_container=predict_container)
    if predict_container:
        column_names = ["Address"]
    elif column_names is None:
        column_names = default_csv_column_name
    elif column_names is not None:
        column_names = column_names
    df = pd.DataFrame(csv_file_content, columns=column_names)
    if reformat_list_fn:
        df.Tags = df.Tags.apply(reformat_list_fn)
    df.to_csv(path, sep=separator, encoding="utf-8")

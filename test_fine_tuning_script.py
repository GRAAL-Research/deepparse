import argparse
import os
import pickle

from poutyne import ReduceLROnPlateau, EarlyStopping

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args):
    address_parser = AddressParser(model=args.model_type, device=0)

    test_container = PickleDatasetContainer(args.test_dataset_path)

    address_parser.test(test_container, batch_size=2048, num_workers=3, logging_path=f"./chekpoints/{args.model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, help='Model type to retrain.')
    parser.add_argument('test_dataset_path', type=str, help='Path to the test dataset.')
    args_parser = parser.parse_args()

    main(args_parser)

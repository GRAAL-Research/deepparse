import argparse

from poutyne import ReduceLROnPlateau, EarlyStopping

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args):
    address_parser = AddressParser(model=args.model_type, device=0)

    container = PickleDatasetContainer(args.dataset_path)

    early_stopping = EarlyStopping(patience=10)
    lr_scheduler = ReduceLROnPlateau()

    address_parser.retrain(container, 0.8, epochs=50, batch_size=1024, num_workers=6,
                           callbacks=[early_stopping, lr_scheduler], logging_path=f"./chekpoints/{args.model_type}")

    address_parser.test(container, batch_size=1024, num_workers=3, logging_path=f"./chekpoints/{args.model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, help='Model type to retrain.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset.')
    args_parser = parser.parse_args()

    main(args_parser)

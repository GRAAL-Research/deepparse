import argparse

from poutyne import ReduceLROnPlateau, EarlyStopping

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args):
    address_parser = AddressParser(model=args.model_type, device=0)

    train_container = PickleDatasetContainer(args.dataset_path)

    early_stopping = EarlyStopping(patience=10)
    lr_scheduler = ReduceLROnPlateau(patience=2)

    address_parser.retrain(train_container, 0.8, epochs=100, batch_size=1024, num_workers=6,
                           callbacks=[early_stopping, lr_scheduler], logging_path=f"./chekpoints/{args.model_type}")

    test_container = PickleDatasetContainer(args.dataset_path)

    address_parser.test(test_container, batch_size=1024, num_workers=3, logging_path=f"./chekpoints/{args.model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, help='Model type to retrain.')
    parser.add_argument('train_dataset_path', type=str, help='Path to the train dataset.')
    parser.add_argument('test_dataset_path', type=str, help='Path to the test dataset.')
    args_parser = parser.parse_args()

    main(args_parser)

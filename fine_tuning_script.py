import argparse

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args):
    address_parser = AddressParser(model=args.model_type, device=0)

    container = PickleDatasetContainer(args.dataset_path)

    # address_parser.retrain(container, 0.8, epochs=1, batch_size=256, num_workers=2)

    address_parser.test(container, batch_size=256, num_workers=2,
                        logging_path=f"/fast/deepparse/retrain-models/eval-before-fine-tuning/{args.model_type}",
                        checkpoint=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, help='Model type to retrain.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset.')
    args_parser = parser.parse_args()

    main(args_parser)

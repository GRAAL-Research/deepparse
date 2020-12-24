# Home to retrain our model
import argparse

from poutyne import StepLR

from deepparse import handle_pre_trained_checkpoint
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def main(args):
    address_parser = AddressParser(model_type=args.model_type, device=0, verbose=False)

    if args.mode in ("train", "both"):
        train_container = PickleDatasetContainer(args.train_dataset_path)

        lr_scheduler = StepLR(step_size=20)

        address_parser.retrain(train_container, 0.8, epochs=100, batch_size=1024, num_workers=6, learning_rate=0.001,
                               callbacks=[lr_scheduler], logging_path=f"./chekpoints/{args.model_type}")

    if args.mode in ("test", "both"):
        test_container = PickleDatasetContainer(args.test_dataset_path)

        if args.mode == "test":
            checkpoint = handle_pre_trained_checkpoint(args.model_type)
        else:
            checkpoint = "best"

        address_parser.test(test_container, batch_size=2048, num_workers=4,
                            logging_path=f"./chekpoints/{args.model_type}", checkpoint=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, help='Model type to retrain.',
                        choices=["fasttext", "bpemb"])
    parser.add_argument('train_dataset_path', type=str, help='Path to the train dataset.')
    parser.add_argument('test_dataset_path', type=str, help='Path to the test dataset.')
    parser.add_argument('mode', type=str, help='Mode type.',
                        choices=["train", "test", "both"])
    args_parser = parser.parse_args()

    main(args_parser)

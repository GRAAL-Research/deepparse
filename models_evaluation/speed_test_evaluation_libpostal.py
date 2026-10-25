import argparse
import os.path

from memory_profiler import profile
from postal.parser import parse_address  # pylint: disable=import-error

from models_evaluation.speed_test_dataset import DEFAULT_NUM_ADDRESSES, build_speed_test_dataset
from models_evaluation.timer.timer import Timer

speed_test_directory = "results/speed_test_results"


@profile
def process_fn(addresses):
    for address in addresses:
        parse_address(address)


def main(args):
    addresses = [address for address, _ in build_speed_test_dataset(num_addresses=args.num_addresses)]
    os.makedirs(speed_test_directory, exist_ok=True)

    timer = Timer()
    with timer:
        process_fn(addresses)

    with open(
        os.path.join(
            speed_test_directory,
            "speed_test_results_with_libpostal.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        print(
            "Temps moyen pour porcess avec Libpostal :",
            timer.elapsed_time / len(addresses),
            file=file,
        )
        print(
            "Temps moyen pour porcess avec Libpostal :",
            timer.elapsed_time / len(addresses),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_addresses",
        type=int,
        default=DEFAULT_NUM_ADDRESSES,
        help="Number of addresses of the speed test dataset (use a small value, e.g. 200, for a quick run).",
    )
    args_parser = parser.parse_args()

    main(args_parser)

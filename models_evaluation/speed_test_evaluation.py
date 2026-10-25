import argparse
import os.path
from statistics import mean

from memory_profiler import profile

from deepparse.parser import AddressParser
from models_evaluation.speed_test_dataset import DEFAULT_NUM_ADDRESSES, build_speed_test_dataset
from models_evaluation.timer.timer import Timer

speed_test_directory = "results/speed_test_results"


@profile
def process_fn(address_parser, addresses, batch_size_arg):
    address_parser(addresses, batch_size=batch_size_arg)


def parse_device(device: str):
    # A GPU device is given as an index (e.g. "0"), while the CPU is given as "cpu".
    return int(device) if device.isdigit() else device


def main(args):
    addresses = [address for address, _ in build_speed_test_dataset(num_addresses=args.num_addresses)]
    os.makedirs(speed_test_directory, exist_ok=True)

    for model in args.model_types:
        for attention_mechanism in [True, False]:
            for device in args.devices:
                times = []
                with open(
                    os.path.join(
                        speed_test_directory,
                        f"speed_test_results_on_{device}_with_{model}_attention-{attention_mechanism}.txt",
                    ),
                    "w",
                    encoding="utf-8",
                ) as file:
                    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                        address_parser = AddressParser(
                            model_type=model,
                            device=device,
                            attention_mechanism=attention_mechanism,
                        )
                        timer = Timer()
                        with timer:
                            process_fn(address_parser, addresses, batch_size)
                        if batch_size == 1:
                            print(
                                "Temps moyen pour batch size avec ",
                                device,
                                "et batch size de ",
                                batch_size,
                                " : ",
                                timer.elapsed_time / len(addresses),
                                file=file,
                            )
                        if batch_size > 1:
                            times.append(timer.elapsed_time / len(addresses))
                    print(
                        "temps moyen pour batch size avec batch size > 1:",
                        mean(times),
                        file=file,
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_addresses",
        type=int,
        default=DEFAULT_NUM_ADDRESSES,
        help="Number of addresses of the speed test dataset (use a small value, e.g. 200, for a quick run).",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["fasttext", "bpemb", "fasttext-light"],
        help="Model types to evaluate.",
    )
    parser.add_argument(
        "--devices",
        type=parse_device,
        nargs="+",
        default=[0, "cpu"],
        help="Devices to evaluate on, either a GPU index (e.g. 0) or 'cpu'.",
    )
    args_parser = parser.parse_args()

    main(args_parser)

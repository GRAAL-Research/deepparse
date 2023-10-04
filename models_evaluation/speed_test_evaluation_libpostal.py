import os.path

from memory_profiler import profile
from postal.parser import parse_address

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer
from models_evaluation.timer.timer import Timer

download_from_public_repository("speed_test_dataset", "./data", "p")

address_container = PickleDatasetContainer("./data/speed_test_dataset.p")
addresses, tags = zip(*address_container)

speed_test_directory = "results/speed_test_results"
os.makedirs(speed_test_directory, exist_ok=True)


@profile
def process_fn():
    for address in addresses:
        parse_address(address)


if __name__ == '__main__':
    timer = Timer()
    with timer:
        process_fn()

    with open(
        os.path.join(
            speed_test_directory,
            f"speed_test_results_with_libpostal.txt",
        ),
        "w",
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

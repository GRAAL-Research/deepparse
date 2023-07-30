import os.path

from postal.parser import parse_address

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer
from models_evaluation.timer.timer import Timer

download_from_public_repository("speed_test_dataset", "./data", "p")

address_container = PickleDatasetContainer("./data/speed_test_dataset.p")
addresses, tags = zip(*address_container)

speed_test_directory = "results/speed_test_results"
os.makedirs(speed_test_directory, exist_ok=True)

with open(
    os.path.join(
        speed_test_directory,
        f"speed_test_results_with_libpostal.txt",
    ),
    "w",
) as file:
    timer = Timer()
    with timer:
        for address in addresses:
            parse_address(address)
    print(
        "Temps moyen pour porcess avec Libpostal :",
        timer.elapsed_time / len(addresses),
        file=file,
    )

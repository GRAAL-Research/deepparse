import os.path
from statistics import mean

from memory_profiler import profile

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser
from models_evaluation.timer.timer import Timer

download_from_public_repository("speed_test_dataset", "./data", "p")

address_container = PickleDatasetContainer("./data/speed_test_dataset.p")
addresses, tags = zip(*address_container)

speed_test_directory = "results/speed_test_results"
os.makedirs(speed_test_directory, exist_ok=True)


@profile
def process_fn(batch_size_arg):
    address_parser(addresses, batch_size=batch_size_arg)


if __name__ == '__main__':
    for model in ["fasttext", "bpemb", "fasttext-light"]:
        for attention_mechanism in [True, False]:
            for device in [0, "cpu"]:
                times = []
                for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                    address_parser = AddressParser(
                        model_type=model,
                        device=device,
                        attention_mechanism=attention_mechanism,
                    )
                    timer = Timer()
                    with timer:
                        process_fn()
                    with open(
                        os.path.join(
                            speed_test_directory,
                            f"speed_test_results_on_{device}_with_{model}_attention-{attention_mechanism}.txt",
                        ),
                        "w",
                    ) as file:
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

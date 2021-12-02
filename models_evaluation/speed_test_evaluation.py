import os.path
import pickle
from statistics import mean

from deepparse import download_from_url
from deepparse.parser import AddressParser
from models_evaluation.timer.timer import Timer

download_from_url("speed_test_dataset", "./data", "p")

addresses = pickle.load(open("./data/speed_test_dataset.p", "rb"))
addresses, tags = zip(*addresses)

speed_test_directory = "results/speed_test_results"
os.makedirs(speed_test_directory, exist_ok=True)

for model in ["fasttext", "bpemb"]:
    for attention_mechanism in [True, False]:
        for device in [0, "cpu"]:
            with open(
                    os.path.join(speed_test_directory,
                                 f"speed_test_results_on_{device}_with_{model}_attention-{attention_mechanism}.txt"),
                    "w") as file:
                times = []
                for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                    address_parser = AddressParser(model_type=model,
                                                   device=device,
                                                   attention_mechanism=attention_mechanism)
                    timer = Timer()
                    with timer:
                        address_parser(addresses, batch_size=batch_size)
                    if batch_size == 1:
                        print("Temps moyen pour batch size avec ",
                              device,
                              "et batch size de ",
                              batch_size,
                              " : ",
                              timer.elapsed_time / len(addresses),
                              file=file)
                    if batch_size > 1:
                        times.append(timer.elapsed_time / len(addresses))
                print("temps moyen pour batch size avec batch size > 1:", mean(times), file=file)

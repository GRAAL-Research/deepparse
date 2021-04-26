import pickle
from statistics import mean

from deepparse import download_from_url
from deepparse.parser import AddressParser
from models_evaluation.timer.timer import Timer

download_from_url("speed_test_dataset", "./data", "p")

addresses = pickle.load(open("./data/speed_test_dataset.p", "rb"))
addresses, tags = zip(*addresses)

for device in [0, "cpu"]:
    times = []
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        address_parser = AddressParser(model_type="fasttext", device=device)
        timer = Timer()
        with timer:
            address_parser(addresses, batch_size=batch_size)
        if batch_size == 1:
            print("temps moyen pour batch size avec ", device, "et batch size de ", batch_size, " : ",
                  timer.elapsed_time / len(addresses))
        if batch_size > 1:
            times.append(timer.elapsed_time / len(addresses))
    print("temps moyen pour batch size avec batch size > 1:", mean(times))

import os

import poutyne
import requests

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


# todo get different data
# First, let's download the dataset
def download_data(saving_dir, data_type):
    """
    Function to download the dataset using data_type to specify if we want the train, valid or test.
    """

    root_url = "https://graal-research.github.io/poutyne-external-assets/tips_and_tricks_assets/{}.p"

    url = root_url.format(data_type)
    r = requests.get(url)
    os.makedirs(saving_dir, exist_ok=True)

    open(os.path.join(saving_dir, f"{data_type}.p"), 'wb').write(r.content)


download_data('./data/', "train")
download_data('./data/', "test")

train_path = "./data/train.p"
test_path = "./data/test.p"

container = PickleDatasetContainer(train_path)

address_parser = AddressParser(model="fasttext", device=0)

# now let's retrain for 5 epoch using a batch size of 128 and the starting default 0.01 learning rate, but using a
# learning rate scheduler to lower the learning rate as we progress.
lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch
address_parser.retrain(container, 0.8, epochs=5, batch_size=128, num_workers=2, callbacks=[lr_scheduler])

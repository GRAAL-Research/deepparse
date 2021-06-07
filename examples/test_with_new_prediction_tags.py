import os
import pickle

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

# First, let's download the test data with new tags from the public repository.
saving_dir = "./data"
file_extension = "p"
test_dataset_name = "test_sample_data_new_prediction_tags"
download_from_url(test_dataset_name, saving_dir, file_extension=file_extension)

# Now let's create a test container.
test_container = PickleDatasetContainer(os.path.join(saving_dir, test_dataset_name + "." + file_extension))

# We will test the fasttext version of our pretrained model. with our own prediction tags dictionary.
# todo fix the example
address_parser = AddressParser(path_to_retrained_model="./checkpoints/retrained_fasttext_address_parser.ckpt", device=0)

address_parser.test(test_container, batch_size=128)

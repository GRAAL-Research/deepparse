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
address_parser = AddressParser(model_type="fasttext", device=0)

# We need a EOS tag in the dictionary. EOS -> End Of Sequence
tag_dictionary = {"ATag": 0, "AnotherTag": 1, "EOS": 2}

# Since we haven't retrained the model, we need to save the dictionary in the directory where we test the model.
# You can also use the default one "./checkpoints"
testing_directory = "test_dir"

os.makedirs(testing_directory, exist_ok=True)

with open(os.path.join(testing_directory, "prediction_tags.p"), "wb") as file:
    pickle.dump(tag_dictionary, file)

address_parser.test(test_container, batch_size=128, checkpoint="fasttext", logging_path=testing_directory)
# For sure the accuracy is necessary low in this example since the tag are now logical.
# But, even with logical one (such as StreetName, StreetNumber, StreetOrientation, or else) the prediction layer
# weights are randomly set.

# pylint: skip-file
###################
"""
IMPORTANT:
THE EXAMPLE IN THIS FILE IS CURRENTLY NOT FUNCTIONAL
BECAUSE THE `download_from_public_repository` FUNCTION
NO LONGER EXISTS. WE HAD TO MAKE A QUICK RELEASE TO
REMEDIATE AN ISSUE IN OUR PREVIOUS STORAGE SOLUTION.
THIS WILL BE FIXED IN A FUTURE RELEASE.

IN THE MEAN TIME IF YOU NEED ANY CLARIFICATION
REGARDING THE PACKAGE PLEASE FEEL FREE TO OPEN AN ISSUE.
"""

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

# Here is an example on how to parse multiple addresses using a URI model place in a S3 Bucket
# First, let's download the train and test data from the public repository.
saving_dir = "./data"
file_extension = "p"
test_dataset_name = "predict"
download_from_public_repository(test_dataset_name, saving_dir, file_extension=file_extension)

#  Now let's load the dataset using one of our dataset container
addresses_to_parse = PickleDatasetContainer("./data/predict.p", is_training_container=False)

# We can sneak peek some addresses
print(addresses_to_parse[:2])

# Let's use the FastText model on a GPU
path_to_your_uri = "s3://<path_to_your_bucket>/fasttext.ckpt"
address_parser = AddressParser(model_type="fasttext", device=0, path_to_retrained_model=path_to_your_uri)

# We can now parse some addresses
parsed_addresses = address_parser(addresses_to_parse[0:300])

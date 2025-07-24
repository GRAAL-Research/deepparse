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
import os

import poutyne

from deepparse import download_from_public_repository
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

# First, let's download the train and test data from the public repository.
saving_dir = "./data"
file_extension = "p"
training_dataset_name = "sample_incomplete_data"
test_dataset_name = "test_sample_data"
download_from_public_repository(training_dataset_name, saving_dir, file_extension=file_extension)
download_from_public_repository(test_dataset_name, saving_dir, file_extension=file_extension)

# Now let's create a training and test container.
training_container = PickleDatasetContainer(os.path.join(saving_dir, training_dataset_name + "." + file_extension))
test_container = PickleDatasetContainer(os.path.join(saving_dir, test_dataset_name + "." + file_extension))

# We will retrain the FastText version of our pretrained model.
path_to_your_uri = "s3://<path_to_your_bucket>/fasttext.ckpt"
address_parser = AddressParser(model_type="fasttext", device=0, path_to_retrained_model=path_to_your_uri)

# Now, let's retrain for 5 epochs using a batch size of 8 since the data is really small for the example.
# Let's start with the default learning rate of 0.01 and use a learning rate scheduler to lower the learning rate
# as we progress.
lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

# The retrained model best checkpoint (ckpt) will be saved in the S3 Bucket <path_to_your_bucket.
address_parser.retrain(
    training_container,
    logging_path="s3://<path_to_your_bucket/",
    train_ratio=0.8,
    epochs=5,
    batch_size=8,
    num_workers=2,
    callbacks=[lr_scheduler],
)

# Now, let's test our fine-tuned model using the best checkpoint (default parameter).
address_parser.test(test_container, batch_size=256)

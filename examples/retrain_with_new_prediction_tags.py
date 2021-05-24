import os
import poutyne
import torch

from deepparse import download_from_url
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

# First, let's download the train and test data with "new tags" from the public repository.
saving_dir = "./data"
file_extension = "p"
training_dataset_name = "sample_incomplete_data_new_prediction_tags"
test_dataset_name = "test_sample_data_new_prediction_tags"
download_from_url(training_dataset_name, saving_dir, file_extension=file_extension)
download_from_url(test_dataset_name, saving_dir, file_extension=file_extension)

# Now let's create a training and test container.
training_container = PickleDatasetContainer(os.path.join(saving_dir, training_dataset_name + "." + file_extension))
test_container = PickleDatasetContainer(os.path.join(saving_dir, test_dataset_name + "." + file_extension))

# We will retrain the fasttext version of our pretrained model.
address_parser = AddressParser(model_type="fasttext", device=0)

# Now let's retrain for 5 epochs using a batch size of 8 since the data is really small for the example.
# Let's start with the default learning rate of 0.01 and use a learning rate scheduler to lower the learning rate
# as we progress.
lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

# We need a EOS tag in the dictionary. EOS -> End Of Sequence
tag_dictionary = {"ATag": 0, "AnotherTag": 1, "EOS": 2}

# The path to save our checkpoints
logging_path = "./checkpoints"

# The checkpoints (ckpt) are saved in the default "./checkpoints" directory.
# address_parser.retrain(training_container, 0.8, epochs=5, batch_size=8, num_workers=2, callbacks=[lr_scheduler],
#                        prediction_tags=tag_dictionary, logging_path=logging_path)

# The retrained model file name are formatted as retrained_{model_type}_address_parser.ckpt
model_path = os.path.join(logging_path, "retrained_fasttext_address_parser.ckpt")
# Now let's test our fine tuned model using the best checkpoint (default parameter).
# address_parser.test(test_container, batch_size=256, model_path=model_path)

data = torch.load(os.path.join(logging_path, "retrained_fasttext_address_parser.ckpt"))

print("a")

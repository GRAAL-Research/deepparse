import poutyne

from deepparse import download_data
from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser

saving_dir = './data/'
training_dataset_name = "sample_noisy_data"
test_dataset_name = "test_sample_data"
download_data(saving_dir, dataset_name=training_dataset_name)
download_data(saving_dir, dataset_name=test_dataset_name)

training_container = PickleDatasetContainer(saving_dir + training_dataset_name + ".p")

address_parser = AddressParser(model="fasttext", device=0)

# now let's retrain for 5 epoch using a batch size of 8 since the data is really small for the example.
# the starting default 0.01 learning rate, but using a learning rate scheduler to lower the learning rate as we progress.
lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

# the ckpt are saved in the default "./chekpoints" directory
address_parser.retrain(training_container, 0.8, epochs=5, batch_size=8, num_workers=2, callbacks=[lr_scheduler])

test_container = PickleDatasetContainer(saving_dir + test_dataset_name + ".p")

# Now let's test our fine tuned model
address_parser.test(test_container, batch_size=256)

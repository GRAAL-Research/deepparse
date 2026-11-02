import poutyne

from deepparse import download_worldwide_addresses_sample
from deepparse.dataset_container import ListDatasetContainer
from deepparse.parser import AddressParser

# First, let's download a small sample of Canadian addresses from the worldwide-addresses dataset
# (https://huggingface.co/datasets/deepparse/worldwide-addresses). The dataset tags are already mapped to the
# deepparse tags. We use the first 1,000 addresses for training and the last 200 for testing.
addresses = download_worldwide_addresses_sample(country_code="ca", num_samples=1200)

# Now let's create a training and test container.
training_container = ListDatasetContainer(addresses[:1000])
test_container = ListDatasetContainer(addresses[1000:])

# We will retrain the FastText version of our pretrained model.
address_parser = AddressParser(model_type="fasttext", device=0)

# Now, let's retrain for 5 epochs using a batch size of 8 since the data is really small for the example.
# Let's start with the default learning rate of 0.01 and use a learning rate scheduler to lower the learning rate
# as we progress.
lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

# The checkpoints (ckpt) are saved in the default "./checkpoints" directory, so if you wish to retrain
# another model (let's say BPEmb), you need to change the `logging_path` directory; otherwise, you will get
# an error when retraining since Poutyne will try to use the last checkpoint.
address_parser.retrain(
    training_container,
    train_ratio=0.8,
    epochs=5,
    batch_size=8,
    num_workers=2,
    callbacks=[lr_scheduler],
)

# Now, let's test our fine-tuned model using the best checkpoint (default parameter).
address_parser.test(test_container, batch_size=256)

# Now let's retrain the FastText version but with an attention mechanism.
address_parser = AddressParser(model_type="fasttext", device=0, attention_mechanism=True)

# Since the previous checkpoints were saved in the default "./checkpoints" directory, we need to use a new one.
# Otherwise, poutyne will try to reload the previous checkpoints, and our model has changed.
address_parser.retrain(
    training_container,
    train_ratio=0.8,
    epochs=5,
    batch_size=8,
    num_workers=2,
    callbacks=[lr_scheduler],
    logging_path="checkpoints_attention",
)

# Now, let's test our fine-tuned model using the best checkpoint (default parameter).
address_parser.test(test_container, batch_size=256)

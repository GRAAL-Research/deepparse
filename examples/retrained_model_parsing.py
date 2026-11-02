import os

from deepparse import download_worldwide_addresses_sample
from deepparse.dataset_container import ListDatasetContainer
from deepparse.parser import AddressParser

# Here is an example on how to parse multiple addresses
# First, let's download a small sample of Canadian addresses from the worldwide-addresses dataset
# (https://huggingface.co/datasets/deepparse/worldwide-addresses). We use the first 500 tagged addresses to create a
# retrained model and the last 300 addresses (without their tags) to parse.
addresses = download_worldwide_addresses_sample(country_code="ca", num_samples=800)

#  Now let's load the datasets using one of our dataset container
training_container = ListDatasetContainer(addresses[:500])
addresses_to_parse = ListDatasetContainer([address for address, _ in addresses[500:]], is_training_container=False)

# Let's create a BPEmb retrained model just for this example (one epoch on our small training sample),
# but you can also use one of yours.
model_saving_dir = "./retrained_models"
address_parser = AddressParser(model_type="bpemb", device=0)
address_parser.retrain(training_container, epochs=1, batch_size=8, num_workers=2, logging_path=model_saving_dir)

# By default, the retrained model is saved as "retrained_{model_type}_address_parser.ckpt" in the logging path.
retrained_model_name = "retrained_bpemb_address_parser"
model_file_extension = "ckpt"

address_parser = AddressParser(
    model_type="bpemb",
    device=0,
    path_to_retrained_model=os.path.join(model_saving_dir, retrained_model_name + "." + model_file_extension),
)

# We can now parse some addresses
parsed_addresses = address_parser(addresses_to_parse[0:300])

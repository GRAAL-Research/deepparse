from deepparse import download_worldwide_addresses_sample
from deepparse.dataset_container import ListDatasetContainer
from deepparse.parser import AddressParser

# Here is an example on how to parse multiple addresses using a URI model place in a S3 Bucket
# First, let's download a small sample of Canadian addresses from the worldwide-addresses dataset
# (https://huggingface.co/datasets/deepparse/worldwide-addresses). We only keep the addresses (not the tags).
addresses = [address for address, _ in download_worldwide_addresses_sample(country_code="ca", num_samples=300)]

#  Now let's load the dataset using one of our dataset container
addresses_to_parse = ListDatasetContainer(addresses, is_training_container=False)

# We can sneak peek some addresses
print(addresses_to_parse[:2])

# Let's use the FastText model on a GPU
path_to_your_uri = "s3://<path_to_your_bucket>/fasttext.ckpt"
address_parser = AddressParser(model_type="fasttext", device=0, path_to_retrained_model=path_to_your_uri)

# We can now parse some addresses
parsed_addresses = address_parser(addresses_to_parse[0:300])

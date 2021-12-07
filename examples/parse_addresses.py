import os
import pickle

import pandas as pd

from deepparse import download_from_url
from deepparse.parser import AddressParser

# Here is an example on how to parse multiple addresses
# First, let's download the train and test data from the public repository.
saving_dir = "./data"
file_extension = "p"
test_dataset_name = "predict"
download_from_url(test_dataset_name, saving_dir, file_extension=file_extension)

# Now let's load the pickled data (in a list format)
with open(os.path.join(saving_dir, test_dataset_name + "." + file_extension), "rb") as f:
    test_data = pickle.load(f)  # a 30000 addresses list

# We can sneak peek some addresses
print(test_data[:2])

# Let's use the BPEmb model on a GPU
address_parser = AddressParser(model_type="bpemb", device=0)

# We can now parse some addresses
parsed_addresses = address_parser(test_data[0:300])

# We can also put our parsed address into a pandas dataframe for analysis
# You can choose the fields to use or use the default one
fields = ["StreetNumber", "StreetName", "Municipality", "Province", "PostalCode"]
parsed_address_data_frame = pd.DataFrame([parsed_address.to_dict(fields=fields) for parsed_address in parsed_addresses],
                                         columns=fields)

import pickle

import numpy as np
import poutyne
from poutyne import set_seeds, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from deepparse.dataset_container import ListDatasetContainer
from deepparse.parser import AddressParser

seed = 42
set_seeds(seed)


def get_train_test_split(dataset):
    train_indices, test_indices = train_test_split(np.arange(len(dataset)), train_size=train_split_percent)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


def change_PoBox_to_POBox(data):
    for key in data.keys():
        for d in data.get(key):
            for idx, t in enumerate(d[1]):
                if t == "PoBox":
                    d[1][idx] = "POBox"


with open("/fast/davidbeauchemin/libpostal_data/train/new_full_data.p", "rb") as file:
    complete_and_incomplete_train_dataset = pickle.load(file)

train_split_percent = 0.8
with open("/fast/davidbeauchemin/libpostal_data/po_boxes/pobox_data_ca.p", "rb") as file:
    ca_po_boxes_complete_and_incomplete = pickle.load(file)
change_PoBox_to_POBox(ca_po_boxes_complete_and_incomplete)

ca_po_boxes_complete_train, ca_po_boxes_complete_test = get_train_test_split(
    ca_po_boxes_complete_and_incomplete.get("complete")
)

ca_po_boxes_incomplete_train, ca_po_boxes_incomplete_test = get_train_test_split(
    ca_po_boxes_complete_and_incomplete.get("incomplete")
)

with open("/fast/davidbeauchemin/libpostal_data/po_boxes/pobox_data_gb.p", "rb") as file:
    gb_po_boxes_complete_and_incomplete = pickle.load(file)
change_PoBox_to_POBox(gb_po_boxes_complete_and_incomplete)

gb_po_boxes_complete_train, gb_po_boxes_complete_test = get_train_test_split(
    gb_po_boxes_complete_and_incomplete.get("complete")
)

gb_po_boxes_incomplete_train, gb_po_boxes_incomplete_test = get_train_test_split(
    gb_po_boxes_complete_and_incomplete.get("incomplete")
)

with open("/fast/davidbeauchemin/libpostal_data/po_boxes/pobox_data_us.p", "rb") as file:
    us_po_boxes_complete_and_incomplete = pickle.load(file)
change_PoBox_to_POBox(us_po_boxes_complete_and_incomplete)

us_po_boxes_complete_train, us_po_boxes_complete_test = get_train_test_split(
    us_po_boxes_complete_and_incomplete.get("complete")
)

us_po_boxes_incomplete_train, us_po_boxes_incomplete_test = get_train_test_split(
    us_po_boxes_complete_and_incomplete.get("incomplete")
)

train_po_boxes_complete_all_country = (
    ca_po_boxes_complete_train + gb_po_boxes_complete_train + us_po_boxes_complete_train
)
train_po_boxes_incomplete_all_country = (
    ca_po_boxes_incomplete_train + gb_po_boxes_incomplete_train + us_po_boxes_incomplete_train
)

address_parser = AddressParser(model_type='fasttext', attention_mechanism=False)

new_prediction_tags = {
    "StreetNumber": 0,
    "StreetName": 1,
    "Unit": 2,
    "Municipality": 3,
    "Province": 4,
    "PostalCode": 5,
    "Orientation": 6,
    "GeneralDelivery": 7,
    "POBox": 8,
    "EOS": 9,  # the 10th is the EOS with idx 9
}

po_boxes_dataset = list(train_po_boxes_complete_all_country) + list(train_po_boxes_incomplete_all_country)
address_parser.retrain(
    ListDatasetContainer(po_boxes_dataset),
    name_of_the_retrain_parser="fasttext_po_boxes_model",
    logging_path="./po_boxes_retrain",
    epochs=10,
    batch_size=8,
    learning_rate=0.0001,
    prediction_tags=new_prediction_tags,
)

ca_complete_res = address_parser.test(ListDatasetContainer(list(ca_po_boxes_complete_test)))
print("Warmup Training CA complete" + " ".join([f"{key}: {value}" for key, value in ca_complete_res.items()]))
ca_incomplete_res = address_parser.test(ListDatasetContainer(list(ca_po_boxes_incomplete_test)))
print("Warmup Training CA incomplete" + " ".join([f"{key}: {value}" for key, value in ca_incomplete_res.items()]))

gb_complete_res = address_parser.test(ListDatasetContainer(list(gb_po_boxes_complete_test)))
print("Warmup Training GB complete" + " ".join([f"{key}: {value}" for key, value in gb_complete_res.items()]))
gb_incomplete_res = address_parser.test(ListDatasetContainer(list(gb_po_boxes_incomplete_test)))
print("Warmup Training GB incomplete" + " ".join([f"{key}: {value}" for key, value in gb_incomplete_res.items()]))

us_complete_res = address_parser.test(ListDatasetContainer(list(us_po_boxes_complete_test)))
print("Warmup Training US complete" + " ".join([f"{key}: {value}" for key, value in us_complete_res.items()]))
us_incomplete_res = address_parser.test(ListDatasetContainer(list(us_po_boxes_incomplete_test)))
print("Warmup Training US incomplete" + " ".join([f"{key}: {value}" for key, value in us_incomplete_res.items()]))

lr_scheduler = poutyne.StepLR(step_size=5, gamma=0.1)  # reduce LR by a factor of 10 each epoch
patience = 11
early_stopping = EarlyStopping(patience=patience)

all_train_data = complete_and_incomplete_train_dataset + po_boxes_dataset

address_parser.retrain(
    ListDatasetContainer(all_train_data),
    name_of_the_retrain_parser="fasttext_po_boxes_model",
    logging_path="./po_boxes_retrain",
    epochs=100,
    batch_size=256,
    learning_rate=0.0001,
    callbacks=[lr_scheduler, early_stopping],
)

ca_complete_res = address_parser.test(ListDatasetContainer(list(ca_po_boxes_complete_test)))
print("Training CA complete" + " ".join([f"{key}: {value}" for key, value in ca_complete_res.items()]))
ca_incomplete_res = address_parser.test(ListDatasetContainer(list(ca_po_boxes_incomplete_test)))
print("Training CA incomplete" + " ".join([f"{key}: {value}" for key, value in ca_incomplete_res.items()]))

gb_complete_res = address_parser.test(ListDatasetContainer(list(gb_po_boxes_complete_test)))
print("Training GB complete" + " ".join([f"{key}: {value}" for key, value in gb_complete_res.items()]))
gb_incomplete_res = address_parser.test(ListDatasetContainer(list(gb_po_boxes_incomplete_test)))
print("Training GB incomplete" + " ".join([f"{key}: {value}" for key, value in gb_incomplete_res.items()]))

us_complete_res = address_parser.test(ListDatasetContainer(list(us_po_boxes_complete_test)))
print("Training US complete" + " ".join([f"{key}: {value}" for key, value in us_complete_res.items()]))
us_incomplete_res = address_parser.test(ListDatasetContainer(list(us_po_boxes_incomplete_test)))
print("Training US incomplete" + " ".join([f"{key}: {value}" for key, value in us_incomplete_res.items()]))

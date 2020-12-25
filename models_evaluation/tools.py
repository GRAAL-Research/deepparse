import os

import pycountry

from deepparse.dataset_container import PickleDatasetContainer
from deepparse.parser import AddressParser


def clean_up_name(country: str) -> str:
    """
    Function to clean up pycountry name
    """
    if "Korea" in country:
        country = "South Korea"
    elif "Russian Federation" in country:
        country = "Russia"
    elif "Venezuela" in country:
        country = "Venezuela"
    elif "Moldova" in country:
        country = "Moldova"
    elif "Bosnia" in country:
        country = "Bosnia"
    return country


# country that we trained on
train_test_files = [
    "br.p", "us.p", "kp.p", "ru.p", "de.p", "fr.p", "nl.p", "ch.p", "fi.p", "es.p", "cz.p", "gb.p", "mx.p", "no.p",
    "ca.p", "it.p", "au.p", "dk.p", "pl.p", "at.p"
]


def train_country_file(file: str) -> bool:
    """
    Validate if a file is a training country (as reference of our article).
    """
    return file in train_test_files


# country that we did not train on
other_test_files = [
    "ie.p", "rs.p", "uz.p", "ua.p", "za.p", "py.p", "gr.p", "dz.p", "by.p", "se.p", "pt.p", "hu.p", "is.p", "co.p",
    "lv.p", "my.p", "ba.p", "in.p", "re.p", "hr.p", "ee.p", "nc.p", "jp.p", "nz.p", "sg.p", "ro.p", "bd.p", "sk.p",
    "ar.p", "kz.p", "ve.p", "id.p", "bg.p", "cy.p", "bm.p", "md.p", "si.p", "lt.p", "ph.p", "be.p", "fo.p"
]


def zero_shot_eval_country_file(file: str) -> bool:
    """
    Validate if a file is a zero shot country (as reference of our article).
    """
    return file in other_test_files


def test_on_country_data(address_parser: AddressParser, file: str, directory_path: str, args) -> tuple:
    """
    Compute the results over a country data.
    """
    country = pycountry.countries.get(alpha_2=file.replace(".p", "").upper()).name
    country = clean_up_name(country)

    print(f"Testing on test files {country}")

    test_file_path = os.path.join(directory_path, file)
    test_container = PickleDatasetContainer(test_file_path)

    results = address_parser.test(test_container,
                                  batch_size=4096,
                                  num_workers=4,
                                  logging_path=f"./chekpoints/{args.model_type}",
                                  checkpoint=args.model_path)
    return results, country

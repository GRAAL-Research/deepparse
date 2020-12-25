def clean_up_name(country):
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


train_test_files = [
    'br.p', 'us.p', 'kp.p', 'ru.p', 'de.p', 'fr.p', 'nl.p', 'ch.p', 'fi.p', 'es.p', 'cz.p', 'gb.p', 'mx.p', 'no.p',
    'ca.p', 'it.p', 'au.p', 'dk.p', 'pl.p', 'at.p'
]


# country that we trained on
def train_country_file(file: str):
    return file in train_test_files


# country that we did not train on
other_test_files = [
    'ie.p', 'rs.p', 'uz.p', 'ua.p', 'za.p', 'py.p', 'gr.p', 'dz.p', 'by.p', 'se.p', 'pt.p', 'hu.p', 'is.p', 'co.p',
    'lv.p', 'my.p', 'ba.p', 'in.p', 're.p', 'hr.p', 'ee.p', 'nc.p', 'jp.p', 'nz.p', 'sg.p', 'ro.p', 'bd.p', 'sk.p',
    'ar.p', 'kz.p', 've.p', 'id.p', 'bg.p', 'cy.p', 'bm.p', 'md.p', 'si.p', 'lt.p', 'ph.p', 'be.p', 'fo.p'
]


def zero_shot_eval_country_file(file: str):
    return file in other_test_files

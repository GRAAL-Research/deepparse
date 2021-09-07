import random
import string

import numpy as np
# import checklist
from checklist.editor import Editor
# from checklist.perturb import Perturb
from checklist.test_types import MFT

editor = Editor(language='french')

street_number_choice = np.random.randint(1, 5000, 10).tolist()
street_name_choice = ["A street name"] * 10
postal_code_choice = []
for i in range(10):
    postal_code_choice.append("".join([
        random.choice(string.ascii_letters).upper(),
        str(np.random.randint(0, 9, 1)[0]),
        random.choice(string.ascii_letters).upper(), " ",
        str(np.random.randint(0, 9, 1)[0]),
        random.choice(string.ascii_letters).upper(),
        str(np.random.randint(0, 9, 1)[0])
    ]))

t = editor.template('{street_number} {street_name} {city} {postal_code}.',
                    street_number=street_number_choice,
                    street_name=street_name_choice,
                    postal_code=postal_code_choice)

test1 = MFT(t.data, labels=1, name='Simple positives', capability='Vocabulary', description='')

from collections import OrderedDict

from torch import save, load

all_layers_params = load("/home/davidbeauchemin/.cache/deepParse/fasttext.ckpt", map_location="cuda:%d" % int(0))

replaced_dict = OrderedDict(
    [(key.replace("model", "lstm"), value) if key.startswith("encoder") else (key, value) for key, value in
     all_layers_params.items()])

save(replaced_dict, open("/home/davidbeauchemin/.cache/deepParse/fasttext.ckpt", "wb"))

dd = load("/home/davidbeauchemin/.cache/deepParse/fasttext.ckpt", map_location="cuda:%d" % int(0))

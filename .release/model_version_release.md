# How to Create a New Model's Version

1. `md5sum <model.ckpt> > model.version`
2. Remove the model.cpkt text in `model.version` file
3. Update latests BPEMB and FastText hash in `tests/test_tools.py`
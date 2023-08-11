## Beta (0.1)

- Initial release of the pretrained models
- Beta version of the API

## 0.1.2

- Modification of assets URL
- Bugfix dictionary use
- Fixed logo
- Fixed typo deepParse -> deepparse
- Fixed authors in setup

## 0.1.3

- Added "contributing to"
- Added fix for comma problem (#56)
- Added content in Address Parser doc for tags definition
- Fixed Pylint bug with PyTorch 1.6
- Fixed `pack_padded` cpu error with PyTorch new release

## 0.1.3.1

- Added a verbose flag to trigger more messages during embedding loading
- Added a verbose flag in model/embeddings download

## 0.2

- Added FastText light model using pymagnitude-light
- Added fine-tuning API to allow fine-tuning of our models (retrain)
- Added a method to test models (test)
- Added metric, loss and dataset container to facilitate fine-tuning training
- Added an example of fine-tuning
- Added way to load retrain model when instantiation of AddressParser

## 0.2.1

- Fixed README

## 0.2.2

- Fixed error with experiment and verbosity as logging trigger on or off

## 0.2.3

- Improved documentation

## 0.3

- Added Libpostal time in the doc
- Documentation improvement
- Added new models evaluation to doc
- Release new models

## 0.3.3

- We have improved the loading of data during prediction. We now use a data loader.
- Updated the performance table of models with the data loader approach.
- Fixed missing import in the parser module.
- Bug fix of the `max_len` for the predictions

## 0.3.4

- Fixed a bug when using batched address. Since we were sorting the address during the forward pass, the output
  prediction tags were not aligned with the supposed parsed address. We have removed the sorting, and now the results
  are more aligned with our research.

## 0.3.5

- Added verbose flag to training and test based on the __init__ of address parser.
- **Breaking change** Since [SciPy 1.6](https://github.com/scipy/scipy/releases/tag/v1.6.0) is released on Python `3.7+`
  , we don't support Python `3.6`.
- Added management for Windows where the FastText model cannot be pickled. On Windows, we use Gensim fasttext model,
  which takes more RAM.

## 0.3.6

- Added a method for a dict conversion of parsed addresses for simpler `Pandas` integration.
- Added examples for parsing addresses and how to convert them into a DataFrame.
- Fixed error with download module.

## 0.4

- Added verbose flag to training and test based on the __init__ of address parser.
- Added a feature to retrain our models with prediction tags dictionary different from the default one.
- Added in-doc code examples.
- Added code examples.
- Small improvement of models implementation.

## 0.4.1

- Added method to specify the format of address components of an `FormattedParsedAddress`. Formatting can specify the
  field separator, the field to be capitalized, and the field to be upper case.

## 0.4.2

- Added `__eq__` method to `FormattedParsedAddress`.
- Improved device management.
- Improved testing.

## 0.4.3

- Fixed typos in one of a file name.
- Added tools to compare addresses (tagged or not).
- Fixed some tests errors.

## 0.4.4

- Fixed import error.

## 0.5

- Added Python 3.9
- Added feature to allow a more flexible way to retrain
- Added a feature to allow retraining of a new seq2seq architecture
- Fixed prediction tags bug when parsing with new tags after retraining

## 0.5.1

- Fixed address_comparer hint typing error
- Fixed some docs errors
- Retrain and test now have more defaults parameters
- Various small code and tests improvements

## 0.6

- Added Attention mechanism models
- Fixed EOS bug

## 0.6.1

- Completed EOS bug fix

## 0.6.2

- Improved (slightly) code speed of data padding method as per PyTorch list or array to Tensor recommendation.
- Improved doc for RuntimeError due to retraining FastText and BPEmb model in the same directory.
- Added error handling RuntimeError when retraining.

## 0.6.3

- Fixed the printing capture to raise the error with Poutyne as of version 1.8. We keep the previous approach as for
  compatibilities with the previous Poutyne version.
- Added a flag to disable or not Tensorboard during retraining.

## 0.6.4

- Bugfix reloading of retrain attention model (PR #110)
- Improve error handling
- Improve doc

## 0.6.5

- Improve error handling of empty data and whitespace-only data.
- Parsing now include two validation on the data quality (not empty and not whitespace only)
- DataContainer now includes data quality test (not empty, not whitespace only, tags not empty, tag the same len as an
  address and data is a list of tuple)
- New CSVDatasetContainer
- DataContainer can now be used to predict using a flag.
- Add a CLI to parse addresses from the command line.

## 0.6.6

- Fixed errors in code examples
- Improved doc of download_from_url
- Improve error management of retrain and test

## 0.6.7

- Fixed errors in data validation
- Improved doc over data validation
- Bugfix data slicing error with data containers
- Add an example on how to use a retrained model

## 0.7

- Improved CLI
- Fixed bug in CLI export dataset
- Improved the doc of the CLI

## 0.7.1

- Hotfix for missing dependency
- Fixed bug with poutyne version handling

## 0.7.2

- Added JSON output support
- Add logging output of parse CLI function
- Hotfix Poutyne version handling

## 0.7.3

- Add freeze layers parameters to freeze layers during retraining

## 0.7.4

- Improve parsed address print
- Bug-fix #124: comma-separated list without whitespace in CSVDatasetContainer
- Add a report when addresses to parse and tags list len differ
- Add an example on how to fine-tune using our CSVDatasetContainer
- Improve data validation for data to parse

## 0.7.5

- Bug-fix Poutyne version handling that causes a print error when a version is 1.11 when retraining
- Add the option to create a named retrain parsing model using by default the architecture setting or using the
  user-given name
- Hot-fix missing raise for DataError validation of address to parse when address is tuple
- Bug-fix handling of string column name for CSVDatasetContainer that raised ValueError
- Improve parse CLI doc and fix error in doc stating JSON format is supported as input data
- Add batch_size to parse CLI
- Add minimum version to Gensim 4.0.0.
- Add a new CLI function, retrain, to retrain from the command line
- Improve doc
- Add `cache_dir` to the BPEmb embedding model and to `AddressParser` to change the embeddings cache directory and
  models weights cache directory
- Change the `saving_dir` argument of `download_fastext_embeddings` and `download_fasttext_magnitude_embeddings`
  function
  to `cache_dir`. `saving_dir` is now deprecated and will be remove in version 0.8.
- Add a new CLI function, test, to test from the command line

## 0.7.6

- Re-release the version 0.7.5 into 0.7.6 due to manipulation error and change in PyPi (now delete does not delete
  release by yank does).

## 0.8

- Improve SEO
- Add cache_dir arg in all CLI functions
- Improve handling of HTTP error in models version verification
- Improve doc
- Add a note for parsing data cleaning (i.e. lowercase, commas removal, and hyphen replacing).
- Add hyphen parsing cleaning step (with a bool flag to activate or not) to improve some country address parsing (
  see [issue 137](https://github.com/GRAAL-Research/deepparse/issues/137)).
- Add ListDatasetContainer for Python list dataset.

## 0.8.1

- Refactored function `download_from_url` to `download_from_public_repository`.
- Add error management when retrain a FastText like model on Windows with a number of workers (`num_workers`) greater
  than 0.
- Improve dev tooling
- Improve CI
- Improve code coverage and pylint
- Add codacy

## 0.8.2

- Bug-fix retrain attention model naming parsing
- Improve error handling when not a DatasetContainer is use in retrain and test API

## 0.8.3

- Add Zenodo DOI

## 0.9

- Add `save_model_weights` method to `AddressParser` to save model weights (PyTorch state dictionary)
- Improve CI
- Added verbose flag for test to activate or deactivate the test verbosity (it override the AddressParser verbosity)
- Add Docker image
- Add `val_dataset` to retrain API to allow the use of a specific val dataset for training
- Remove deprecated `download_from_url` function
- Remove deprecated `dataset_container` argument
- Fixed error and docs
- Added the UK retrain example

## 0.9.1

- Hotfix cli.download_model attention model bug

## 0.9.2

- Improve Deepparse server error handling and error output.
- Remove deprecated argument `saving_dir` in `download_fasttext_magnitude_embeddings`
  and `download_fasttext_embeddings` functions.
- Add offline argument to remove verification of the latest version.
- Bug-fix cache handling in download model.
- Add `download_models` CLI function.
- [Temporary hot-fix BPEmb SSL certificate error](https://github.com/GRAAL-Research/deepparse/issues/156).

## 0.9.3

- Improve error handling.
- Bug-fix FastText error not handled in test API.
- Add feature to allow new_prediction_tags to retrain CLI.

## 0.9.4

- Improve codebase.

## 0.9.5

- Fixed tags converter bug with data processor.

## 0.9.6

- Add Python 3.11.
- Add pre-processor when parsing addresses.
- Add `pin_memory=True` when using a CUDA device to increase performance as suggested
  by [Torch documentation](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
- Add `torch.no_grad()` context manager in `__call__()` to increase performance.
- Reduce memory swap between CPU and GPU by instantiating Tensor directly on the GPU device.
- Improve some Warnings clarity (i.e. category and message).
- Bug-fix MacOS multiprocessing. It was impossible to use in multiprocess since we were not testing whether torch
  multiprocess was set properly. Now, we set it properly and raise a warning instead of an error.
- Drop Python 3.7 support since newer Python versions are faster
  and [Torch 2.0 does not support Python 3.7](https://dev-discuss.pytorch.org/t/dropping-support-for-cuda-11-6-and-python-3-7-from-pytorch-2-0-release/1021).
- Improve error handling with wrong checkpoint loading in AddressParser retrain_path use.
- Add `torch.compile` integration to improve performance (Torch 1.x still supported) with `mode="reduce-overhead"` as
  suggested in the [documentation](https://pytorch.org/tutorials//intermediate/torch_compile_tutorial.html). It
  increases the performance by about 1/100.

## 0.9.7

- New models release with more meta-data.
- Add a feature to use an AddressParser from a URI.
- Add a feature to upload the trained model to a URI.
- Add an example of how to use URI for parsing from and uploading to.
- Improve error handling of `path_to_retrain_model`.
- Bug-fix pre-processor error.
- Add verbose override and improve verbosity handling in retrain.
- Bug-fix the broken FastText installation using `fasttext-wheel` instead of `fasttext` (
  see [here](https://github.com/facebookresearch/fastText/issues/512#issuecomment-1534519551)
  and [here](https://github.com/facebookresearch/fastText/pull/1292)).

## 0.9.8

- Hot-Fix wheel install (See [issue 196](https://github.com/GRAAL-Research/deepparse/issues/196)).
- Starting now, we also include model weights release in the GitHub release.

## 0.9.9

- Add version to Seq2Seq and AddressParser.
- Add a Deepparse as an API using FastAPI.
- Add a Dockerfile and a `docker-compose.yml` to build a Docker container for the API.
- Bug-fix the default pre-processors that were not all apply but only the last one.

## dev
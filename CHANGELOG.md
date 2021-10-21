## Beta (0.1)

- Initial release of the pre-trained models
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

- Added a verbose flag to trigger more message during embedding loading
- Added a verbose flag in model/embeddings download

## 0.2

- Added Fasttext light model using pymagnitude-light
- Added finetuning API to allow finetuning of our models (retrain)
- Added a method to test models (test)
- Added metric, loss and dataset container to facilitate finetuning training
- Added an example of finetuning
- Added way to load retrain model when instantiation of AddressParser

## 0.2.1

- Fixed README

## 0.2.2

- Fixed error with experiment and verbosity as logging trigger on or off

## 0.2.3

- Improved documentation

## 0.3

- Added Libpostal time in doc
- Documentation improvement
- Added new models evaluation to doc
- Release new models

## 0.3.3

- We have improved the loading of data during prediction. We now use a data loader.
- Updated the performance table of model with the data loader approach.
- Fixed missing import in the parser module.
- Bug fix of the `max_len` for the predictions

## 0.3.4

- Fixed a bug when use batched address. Since we were sorting the address during the forward pass, the output prediction
  tags were not aligned with the supposed parsed address. We have removed the sorting, and now the results are more
  aligned with our research.

## 0.3.5

- Added verbose flag to training and test base on the __init__ of address parser.
- **Breaking change** Since [SciPy 1.6](https://github.com/scipy/scipy/releases/tag/v1.6.0) is released on Python `3.7+`
  , we don't support Python `3.6`.
- Added management for Windows where the FastText model cannot be pickled. On Windows, we use Gensim fasttext model,
  which takes more RAM.

## 0.3.6

- Added a method for a dict conversion of parsed addresses for simpler `Pandas` integration.
- Added examples for parsing addresses and how to convert them into a DataFrame.
- Fixed error with download module.

## 0.4

- Added verbose flag to training and test base on the __init__ of address parser.
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

- Fixed typos in one of file name.
- Added tools to compare addresses (tagged or not).
- Fixed some tests errors.

## 0.4.4

- Fixed import error.

## 0.5

- Added Python 3.9
- Added feature to allow a more flexible way to retrain
- Added a feature to allow retrain of a new seq2seq architecture
- Fixed prediction tags bug when parsing with new tags after retraining
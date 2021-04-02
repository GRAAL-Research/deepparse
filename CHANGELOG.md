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
- Fixed missing import in the parser module.
- Bug fix of the `max_len` for the predictions

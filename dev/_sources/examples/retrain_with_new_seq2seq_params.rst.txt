.. role:: hidden
    :class: hidden-section

Retrain With New Seq2Seq Parameters
***********************************

.. code-block:: python

    import poutyne

    from deepparse import download_worldwide_addresses_sample
    from deepparse.dataset_container import ListDatasetContainer
    from deepparse.parser import AddressParser

First, let's download a small sample of Canadian addresses from the
`worldwide-addresses dataset <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_ and map its tags to our
new tags, ``"new tags"``: ``"ATag"`` for the street part of the address (street number, street name and unit) and
``"AnotherTag"`` for the rest. We use the first 1,000 addresses for training and the last 200 for testing.

.. code-block:: python

    street_tags = ("StreetNumber", "StreetName", "Unit")
    addresses = [
        (address, ["ATag" if tag in street_tags else "AnotherTag" for tag in tags])
        for address, tags in download_worldwide_addresses_sample(country_code="ca", num_samples=1200)
    ]

Now let's create a training and test container.

.. code-block:: python

    training_container = ListDatasetContainer(addresses[:1000])
    test_container = ListDatasetContainer(addresses[1000:])

We will retrain the ``FastText`` version of our pretrained model.

.. code-block:: python

    model = "fasttext"
    address_parser = AddressParser(model_type=model, device=0)

Now, let's retrain for ``5`` epochs using a batch size of ``8`` since the data is really small for the example.
Let's start with the default learning rate of ``0.01`` and use a learning rate scheduler to lower the learning rate as we progress.

.. code-block:: python

    # Reduce LR by a factor of 10 each epoch
    lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)

    # We need a EOS tag in the dictionary. EOS -> End Of Sequence
    tag_dictionary = {"ATag": 0, "AnotherTag": 1, "EOS": 2}

    # The path to save our checkpoints
    logging_path = "./checkpoints"

    # The new seq2seq params settings using smaller hidden size
    # See the documentation for the list of tunable seq2seq parameters
    seq2seq_params = {
        "encoder_hidden_size": 512,
        "decoder_hidden_size": 512
    }

    address_parser.retrain(training_container,
                           train_ratio=0.8,
                           epochs=5,
                           batch_size=8,
                           num_workers=2,
                           callbacks=[lr_scheduler],
                           prediction_tags=tag_dictionary,
                           logging_path=logging_path,
                           seq2seq_params=seq2seq_params)

Now let's test our fine-tuned model using the best checkpoint (default parameter).

.. code-block:: python

    address_parser.test(test_container, batch_size=256)

.. role:: hidden
    :class: hidden-section

Retrain an Attention Mechanism Model
************************************

.. code-block:: python

    import poutyne

    from deepparse import download_worldwide_addresses_sample
    from deepparse.dataset_container import ListDatasetContainer
    from deepparse.parser import AddressParser

First, let's download a small sample of Canadian addresses from the
`worldwide-addresses dataset <https://huggingface.co/datasets/deepparse/worldwide-addresses>`_. The dataset tags are
already mapped to the deepparse tags. We use the first 1,000 addresses for training and the last 200 for testing.

.. code-block:: python

    addresses = download_worldwide_addresses_sample(country_code="ca", num_samples=1200)

Now let's create a training and test container.

.. code-block:: python

    training_container = ListDatasetContainer(addresses[:1000])
    test_container = ListDatasetContainer(addresses[1000:])

We will retrain the ``FastText`` attention version of our pretrained model.

.. code-block:: python

    model = "bpemb"
    address_parser = AddressParser(model_type=model, device=0, attention_mechanism=True)

Now, let's retrain for ``5`` epochs using a batch size of ``8`` since the data is really small for the example.
Let's start with the default learning rate of ``0.01`` and use a learning rate scheduler to lower the learning rate
as we progress.

.. code-block:: python

    lr_scheduler = poutyne.StepLR(step_size=1, gamma=0.1)  # reduce LR by a factor of 10 each epoch

    logging_path = "./checkpoints"

    address_parser.retrain(
        training_container,
        train_ratio=0.8,
        epochs=5,
        batch_size=8,
        num_workers=2,
        callbacks=[lr_scheduler],
        logging_path=logging_path
    )

Now, let's test our fine-tuned model using the best checkpoint (default parameter).

.. code-block:: python

    address_parser.test(test_container, batch_size=256)

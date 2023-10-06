.. role:: hidden
    :class: hidden-section

Training guide
==============

In addition to parsing addresses out-of-the-box, Deepparse allows you to retrain the pre-trained models to fit your own data and use-cases. 
In the world of machine learning, this is what's refered to as `fine-tuning`, which can make it easier to obtain well-performing
models more efficiently and with less data.

Since the process of fine-tuning models can be tricky, this section of the documentation provides some guidelines and insights that may 
be useful in order to successfully adapt our models. See :ref:`fine_tuning` to view a coding example of
how to retrain our models.

.. note::
   We provide practical recommendations for the fine-tuning process, but you may have to try multiple retraining configurations to achieve an optimal result. If you are having difficulty adapting our models to your use-case, feel free to
   open an issue on the deepparse `Github <https://github.com/GRAAL-Research/deepparse/issues>`_.

A few use-cases may lead you to want to retrain Deepparse's models. Whether you wish to obtain a better 
performance on a single or multiple countries that our models weren't trained on, or your data and address schemes require a more complex 
architecture, or the tag structure of your dataset is different from ours; deepparse's retraining features accomodate all these use-cases, and more.

In practice, our models were trained on 20 countries and demonstrated very accurate results on all of them, so unless you wish to predict 
different tags (e.g: StreetNumber...) we advise you to use our models without retraining. Also, if you wish to retrain 
our models to obtain a better performance on countries outside of the 20 used in the original training set, you can take a look 
at `our dataset <https://github.com/GRAAL-Research/deepparse-address-data>`_ which includes an additional 41 countries that were only used for testing.

There are two main concerns to keep in mind when fine-tuning a model: the model's convergence (i.e: its ability to actually learn from the new data) 
and the possibility of `catastrophic forgetting` (i.e: losing the model's previous knowledge after training on the new data).

Learning successfully
*********************

Making a model converge is as much an art as it is a science since it often requires quite a bit of experimentation and parameter tuning. In the case 
of fine-tuning, the models have already developed a base knowledge of the task that they were trained on, which gives them an edge.
This is especially true in the case of deepparse, since the task you are fine-tuning on remains exactly the same (i.e: parsing addresses). 
However, there are a couple of points to consider in order to obtain favourable results:

- **Make sure you have enough data**: deep learning models are notorious for being pretty data hungry, so unless you have enough data the models 
  are going have a hard time learning. Since deepparse's models have already been trained on a few million addresses, the need for data is mitigated when it come to fine-tuning. However, it is recommended to at least have a few thousand examples per new country when retraining.

- **Prepare your dataset**: once you are done pre-processing your dataset, you must convert it to a format which can be loaded into 
  a :class:`~deepparse.dataset_container.DatasetContainer`. See the :ref:`dataset_container` section for more details.
  Also, make sure to keep a portion of your data apart to test the performance of your retrained models.

- **Use a proper learning rate**: if you are unfamiliar with gradient descent and neural network optimisation you probably don't know what 
  a `learning rate` is. But have no fear, you do not need a Phd to retrain deepparse's models. All you need to understand is that a learning rate 
  is a value that guides the training process. When it comes to fine-tuning, it is recommended to use a learning rate lower than the one use for the first 
  training, in this case we recommend using a learning rate lower than 0.1. This parameter can be changed in the :meth:`~deepparse.parser.AddressParser.retrain` method.

- **Train for long enough**: deepparse's models are based on the LSTM neural network architecture, which may require a few more training epochs 
  than recent architectures when it comes to fine-tuning. The actual number of epochs would depend on the use-case, but it is
  important to allow the models to train long enough. Perhaps start somewhere between 5 and 10 epochs and increase the number of epochs if needed.

- **Use a GPU**: this is not required for retraining but it is highly recommended to use a GPU if your device has one in order to speed up the 
  training process. This can be specified in the :class:`~deepparse.parser.AddressParser` constructor.

Do not forget!
**************

As mentionned above, catastrophic forgetting can happen when fine-tuning machine learning models. This is because the models' internal parameters are 
modified to accomodate the new task/data which can impact their ability to be appropriate for the previous task/data.

There are many fancy ways to mitigate catastrophic forgetting when fine-tuning models, but given the task and data that deepparse handles, we simply 
recommend to include some of the previous data when constructing your retraining dataset. The amount
of addresses to keep would vary depending on the number of new addresses, but somewhere between 1% and 10% would be a good start.

Another approach that can help reduce the effect of forgetting is freezing part of the model. Check out 
the :meth:`~deepparse.parser.AddressParser.retrain` method for more details on how to freeze different parts of our models during retraining.

.. note::
   If you're only interested in the models' performance on the new data, you should not concern yourself with catastrophic forgetting.

Modifying the architecture
**************************

The :meth:`~deepparse.parser.AddressParser.retrain` method allows you to change the architecture of the models using the ``seq2seq_params`` 
argument. This can be useful if you need a more complex model or a lighter model for example. However, if you
choose to change the models' architecture, you will end up with a completely new model that will be retrained from scratch. This 
means that all the previous knowledge that the initial model had will disapear.

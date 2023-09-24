.. role:: hidden
    :class: hidden-section

Models Fine-tuning Guidelines
=============================

Here is some guidelines for fine-tuning our models to our specific needs. To do so you will either need our
dataset (URL) or need new annotated data. Our model were trained on a total of 2 millions address, namely
100,000 address for 20 countries. Our models show great performance on never seen country address and new language
(see our articles for more detail URL).

If you want to improve performance on only one country, we recommend using our dataset (if applicable) or your dataset.
Use a 80-20 train-test split for a couple of epoch (5 to 10 is usually enough).
If model size is a bottleneck, you could also reduce the model size and test when performance decrease vis-a-vis model size
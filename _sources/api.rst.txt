.. role:: hidden
    :class: hidden-section

Parse Address With Our Out-Of-The-Box API
=========================================

We also offer an out-of-the-box REST API to parse addresses using FastAPI.

Installation
************

First, ensure you have Docker Engine and Docker Compose installed on your machine.
If not, you can install them using the following documentation in the following order:

1. `Docker Engine <https://docs.docker.com/engine/install/>`_
2. `Docker Compose <https://docs.docker.com/compose/install/>`_

Once you have Docker Engine and Docker Compose installed, you can run the following command to start the FastAPI application:

.. code-block:: sh

    docker compose up app

Sentry
******

Also, you can monitor your application usage with `Sentry <https://sentry.io>`_ by setting the environment variable ``SENTRY_DSN`` to your Sentry project
DSN. There is an example of the ``.env`` file in the project's root named ``.env_example``. You can copy it using the following command:

.. code-block:: sh

    cp .env_example .env

Request Examples
----------------

Once the application is up and running and port ``8000`` is exported on your ``localhost``, you can send a request with one
of the following methods:

cURL POST request
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    curl -X POST --location "http://127.0.0.1:8000/parse/bpemb-attention" --http1.1 \
        -H "Host: 127.0.0.1:8000" \
        -H "Content-Type: application/json" \
        -d "[
              {\"raw\": \"350 rue des Lilas Ouest Quebec city Quebec G1L 1B6\"},
              {\"raw\": \"2325 Rue de l'Université, Québec, QC G1V 0A6\"}
            ]"

Python POST request
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests

    url = 'http://localhost:8000/parse/bpemb'
    addresses = [
        {"raw": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
        {"raw": "2325 Rue de l'Université, Québec, QC G1V 0A6"}
        ]

    response = requests.post(url, json=addresses)
    parsed_addresses = response.json()
    print(parsed_addresses)

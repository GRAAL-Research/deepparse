.. role:: hidden
    :class: hidden-section

Parse Address With Our Out-Of-The-Box REST API
==============================================

We also offer an out-of-the-box RESTAPI to parse addresses using FastAPI.

Installation
************

First, ensure that you have Docker Engine and Docker Compose installed on your machine.
If not, you can install them using the following documentations in the following order:

1. `Docker Engine <https://docs.docker.com/engine/install/>`_
2. `Docker Compose <https://docs.docker.com/compose/install/>`_


Most parameter values are hardcoded directly in the ``docker-compose.yml`` file, but there is a ``.env`` file that contains secret or personnal values. There is an example of the ``.env`` file in the project's root named ``.env_example``. You can copy it using the following command:

.. code-block:: sh

    cp .env_example .env

You can then adjust the values to your needs.

REST API
********

Once you have Docker Engine and Docker Compose installed and you created your ``.env`` file, you can serve the API service using this command:

.. code-block:: sh

    docker compose up app

Once the API is launched, you can navigate to localhost:8081/docs to see OpenAPI documentation of the FastAPI endpoints. This is the default address used in the project, you can change it in the ``docker-compose.yml`` file.

Make your API secure with https-portal
**************************************

you can run the following command to start the `webserver <https://github.com/SteveLTN/https-portal>`_ service that provides SSL security to the endpoint as well as automatic renewal of certificates, it uses NGINX in the background, do not worry, it has a MIT license. Do not forget to add your domain name in the ``.env`` file.:

.. code-block:: sh

    docker compose up webserver

Sentry
******

Also, you can monitor your application usage with `Sentry <https://sentry.io>`_ by setting the environment variable  ``SENTRY_DSN`` to your Sentry's project DSN. 

If you do not have a Sentry account, you can create one `here <https://sentry.io/signup/>`_.

If you do not want to use Sentry, you can just remove the ``SENTRY_DSN`` environment variable from the ``.env`` file or set it to an empty string, The api will run without any problem if Sentry is not set.

Request Examples
----------------

Once the application is up and running and your selected port is exported on your host, you can send a request with one
of the following methods, the host is ``localhost`` and the selected port is ``8081``:

cURL POST request
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    curl -X POST --location "http://localhost:8081/parse/bpemb-attention" --http1.1 \
        -H "Host: localhost:8081" \
        -H "Content-Type: application/json" \
        -d "[
              {\"raw\": \"350 rue des Lilas Ouest Quebec city Quebec G1L 1B6\"},
              {\"raw\": \"2325 Rue de l'Université, Québec, QC G1V 0A6\"}
            ]"

Python POST request
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import requests

    url = 'http://localhost:8081/parse/bpemb'
    addresses = [
        {"raw": "350 rue des Lilas Ouest Quebec city Quebec G1L 1B6"},
        {"raw": "2325 Rue de l'Université, Québec, QC G1V 0A6"}
        ]

    response = requests.post(url, json=addresses)
    parsed_addresses = response.json()
    print(parsed_addresses)
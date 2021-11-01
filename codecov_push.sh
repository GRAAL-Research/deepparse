#!/bin/sh

pip install pytest-cov
pip install -Ur tests/requirements.txt
pip install -Ur requirements.txt
pytest --cov=./deepparse --cov-report=xml --cov-config=.coveragerc
./codecov -f coverage.xml -n unittest-integrationtest -t $CODECOVKEYDEEPPARSE

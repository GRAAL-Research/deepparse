#!/bin/sh

pip install pytest-cov
pytest --cov=./deepparse --cov-report=xml --cov-config=.coveragerc
./codecov -f coverage.xml -n unittest-integrationtest -t $CODECOVKEYDEEPPARSE

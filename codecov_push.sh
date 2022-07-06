#!/bin/sh

# We do test on Python 3.10

# Create a new Python env for Deepparse tests and activate it
conda create --name deepparse_pytest_3_10 python=3.10 -y --force
conda activate deepparse_pytest_3_10

# Install dependencies
pip install -Ur tests/requirements.txt
pip install -Ur requirements.txt

# Run pytest from conda env
echo "Running test in Conda - No update"
# --live-stream is to remove the Conda capture of the stream
conda run -n deepparse_pytest_3_10 --live-stream pytest --cov ./deepparse --cov-report html --cov-report xml --cov-config=.coveragerc

# Push the coverage file
./codecov -f coverage.xml -n unittest-integrationtest -t $CODECOVKEYDEEPPARSE

# close conda env
conda deactivate

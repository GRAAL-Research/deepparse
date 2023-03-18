#!/bin/sh
# To run using ZSH: zsh -i ./codecov_push.sh
# To run using bash: bash -i ./codecov_push.sh

# We do test on Python 3.11 (latest Python version)

# Create a new Python env for Deepparse tests and activate it
conda create --name deepparse_pytest_3_11 python=3.11 -y --force
conda activate deepparse_pytest_3_11

# Install dependencies
pip install -Ur tests/requirements.txt
pip install -Ur requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda*****"
# --live-stream is to remove the Conda capture of the stream
conda run -n deepparse_pytest_3_11 --live-stream pytest --cov ./deepparse --cov-report html --cov-report xml --cov-config=.coveragerc

# Push the coverage file
./codecov -f coverage.xml -n unittest-integrationtest -t $CODECOVKEYDEEPPARSE

# close conda env
conda deactivate

conda env remove -n deepparse_pytest_3_11

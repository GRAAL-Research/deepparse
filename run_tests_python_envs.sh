#!/bin/sh

# We test on supported Python version, namely, 3.7, 3.8, 3.9 and 3.10
echo "*****Starting of testing Deepparse on Python version 3.7, 3.8, 3.9, 3.10*****"

# We export the reports into a directory. But to do so, we need to move into that directory
# and run pytest from there
mkdir -p export_html_reports
cd ./export_html_reports

# Create a new Python env 3.7
conda create --name deepparse_pytest_3_7 python=3.7 -y --force
conda activate deepparse_pytest_3_7

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.7*****"
conda run -n deepparse_pytest_3_7 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_7 --cov-report xml:export_xml_report_3_7.xml --cov-config=.coveragerc ../tests

# close conda env
conda deactivate

# Create a new Python env 3.8
conda create --name deepparse_pytest_3_8 python=3.8 -y --force
conda activate deepparse_pytest_3_8

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.8*****"
conda run -n deepparse_pytest_3_8 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_8 --cov-report xml:export_xml_report_3_8.xml --cov-config=.coveragerc ../tests

# close conda env
conda deactivate

# Create a new Python env 3.9
conda create --name deepparse_pytest_3_9 python=3.9 -y --force
conda activate deepparse_pytest_3_9

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.9*****"
conda run -n deepparse_pytest_3_9 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_9 --cov-report xml:export_xml_report_3_9.xml --cov-config=.coveragerc ../tests

# close conda env
conda deactivate

# Create a new Python env 3.10
conda create --name deepparse_pytest_3_10 python=3.10 -y --force
conda activate deepparse_pytest_3_10

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.10*****"
conda run -n deepparse_pytest_3_10 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_10 --cov-report xml:export_xml_report_3_10.xml --cov-config=.coveragerc ../tests

# close conda env
conda deactivate

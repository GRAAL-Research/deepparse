#!/bin/sh

# We test on Deepparse supported Python versions, namely, 3.7, 3.8, 3.9 and 3.10
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

if [ $? -eq 0 ]; then
  python3_7_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_7

# Create a new Python env 3.8
conda create --name deepparse_pytest_3_8 python=3.8 -y --force
conda activate deepparse_pytest_3_8

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.8*****"
conda run -n deepparse_pytest_3_8 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_8 --cov-report xml:export_xml_report_3_8.xml --cov-config=.coveragerc ../tests

if [ $? -eq 0 ]; then
  python3_8_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_8

# Create a new Python env 3.9
conda create --name deepparse_pytest_3_9 python=3.9 -y --force
conda activate deepparse_pytest_3_9

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.9*****"
conda run -n deepparse_pytest_3_9 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_9 --cov-report xml:export_xml_report_3_9.xml --cov-config=.coveragerc ../tests

if [ $? -eq 0 ]; then
  python3_9_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_9

# Create a new Python env 3.10
conda create --name deepparse_pytest_3_10 python=3.10 -y --force
conda activate deepparse_pytest_3_10

# Install dependencies
pip install -Ur ../tests/requirements.txt
pip install -Ur ../requirements.txt

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.10*****"
conda run -n deepparse_pytest_3_10 --live-stream pytest --cov ../deepparse --cov-report html:html_report_3_10 --cov-report xml:export_xml_report_3_10.xml --cov-config=.coveragerc ../tests

if [ $? -eq 0 ]; then
  python3_10_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_10

# All tests env print
echo "*****The results of the tests are:"
return_status=0

if [ $python3_7_tests_res -eq 1 ]; then
  echo "Success for Python 3.7"
else
  return_status=1
  echo "Fail for Python 3.7"
fi

if [ $python3_8_tests_res -eq 1 ]; then
  echo "Success for Python 3.8"
else
  return_status=1
  echo "Fail for Python 3.8"
fi

if [ $python3_9_tests_res -eq 1 ]; then
  echo "Success for Python 3.9"
else
  return_status=1
  echo "Fail for Python 3.9"
fi

if [ $python3_10_tests_res -eq 1 ]; then
  echo "Success for Python 3.10"
else
  return_status=1
  echo "Fail for Python 3.10"
fi

if [ $return_status -eq 1]; then
  exit 1
else
  exit 0
fi

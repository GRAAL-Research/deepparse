#!/bin/sh
# To run using ZSH: zsh -i ./codecov_push.sh
# To run using bash: bash -i ./codecov_push.sh

# We test on Deepparse supported Python versions, namely, 3.8, 3.9, 3.10 and 3.11
echo "*****Starting of testing Deepparse on Python version 3.8, 3.9, 3.10, 3.11*****"

# We export the reports into a directory. But to do so, we need to move into that directory
# and run pytest from there
mkdir -p export_html_reports
cd ./export_html_reports

# Create a new Python env 3.8
conda create --name deepparse_pytest_3_8 python=3.8 -y --force
conda activate deepparse_pytest_3_8

# Install dependencies
pip install -e '..[all]'

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.8*****"
conda run -n deepparse_pytest_3_8 --live-stream pytest -o env="TEST_LEVEL=all" --cov ../deepparse --cov-report html:html_report_3_8 --cov-report xml:export_xml_report_3_8.xml --cov-config=.coveragerc ../tests

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
pip install -e '..[all]'

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.9*****"
conda run -n deepparse_pytest_3_9 --live-stream pytest -o env="TEST_LEVEL=all" --cov ../deepparse --cov-report html:html_report_3_9 --cov-report xml:export_xml_report_3_9.xml --cov-config=.coveragerc ../tests

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
pip install -e '..[all]'

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.10*****"
conda run -n deepparse_pytest_3_10 --live-stream pytest -o env="TEST_LEVEL=all" --cov ../deepparse --cov-report html:html_report_3_10 --cov-report xml:export_xml_report_3_10.xml --cov-config=.coveragerc ../tests

if [ $? -eq 0 ]; then
  python3_10_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_11

# Create a new Python env 3.11
conda create --name deepparse_pytest_3_11 python=3.11 -y --force
conda activate deepparse_pytest_3_11

# Install dependencies
pip install -e '..[all]'

# Run pytest from conda env
echo "*****Running test in Conda Python version 3.11*****"
conda run -n deepparse_pytest_3_11 --live-stream pytest -o env="TEST_LEVEL=all" --cov ../deepparse --cov-report html:html_report_3_11 --cov-report xml:export_xml_report_3_11.xml --cov-config=.coveragerc ../tests

if [ $? -eq 0 ]; then
  python3_11_tests_res=1
fi

# close conda env
conda deactivate

# Cleanup the conda env
conda env remove -n deepparse_pytest_3_11

# All tests env print
echo "*****The results of the tests are:"
return_status=0

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

if [ $python3_11_tests_res -eq 1 ]; then
  echo "Success for Python 3.11"
else
  return_status=1
  echo "Fail for Python 3.11"
fi

if [ $return_status -eq 1 ]; then
  exit 1
else
  exit 0
fi

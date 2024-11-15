#!/bin/bash

set -eu -o pipefail # fail on error and report it, debug all lines

sudo -n true
test $? -eq 0 || exit 1 "you should have sudo privilege to run this s"

container_name='NeuralNetworkOnWeatherPrediction'

CONTAINER_ID=$(docker ps -q --filter "name=${container_name}")

# PEP8 report
docker exec $CONTAINER_ID sh -c 'command -v pep8 >/dev/null 2>&1 || { apt-get update && apt-get install -y python3-pep8; } && pep8 /path/to/your/directory'

# change to 'test_*' for full output
# need to change test_all to test_*
docker exec $CONTAINER_ID python3 -m unittest discover -s 'testing/' -v -p 'test_all.py'

# Install coverage package if not already installed
docker exec $CONTAINER_ID sh -c 'command -v coverage >/dev/null 2>&1 || { echo >&2 "Coverage is not installed. Installing..."; apt-get update && apt-get install -y python3-coverage; }'

# this is a coverage report
docker exec $CONTAINER_ID coverage html -d coverage_report

# Install pip if not already installed and generate requirements.txt
docker exec $CONTAINER_ID sh -c 'command -v pip >/dev/null 2>&1 || { echo >&2 "pip is not installed. Installing..."; apt-get update && apt-get install -y python3-pip; };'

# Install pep8 or pycodestyle if not already installed
docker exec $CONTAINER_ID sh -c 'command -v pep8 >/dev/null 2>&1 || command -v pycodestyle >/dev/null 2>&1 || { echo >&2 "pep8 or pycodestyle is not installed. Installing..."; apt-get update && apt-get install -y python3-pep8 || apt-get install -y python3-pycodestyle; };'

echo "We are here"
docker exec $CONTAINER_ID sh -c 'command -v python3 /usr/lib/python3/dist-packages/pep8.py /anemometer | tee pep8_report.txt'

# Generate requirements.txt
docker exec $CONTAINER_ID pip freeze | tee requirements.txt
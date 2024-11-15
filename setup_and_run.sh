#!/bin/bash

# Check if a parameter is passed to the script
if [ -z "$1" ]; then
  echo "Error: No parameter provided. Please provide a string parameter."
  exit 1
fi

# Capture the passed parameter
PARAM="$1"

# Run orchestration.py with the passed parameter as an environment variable
echo "Running orchestration.py with parameter: $PARAM"
python orchestration.py "$PARAM"

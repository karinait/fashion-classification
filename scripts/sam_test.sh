#!/bin/bash

# Navigate to the directory where the .aws-build/template.yaml is located
cd ../fashion-classifier || exit

# Run the SAM local invoke command
sam local invoke -t .aws-build/template.yaml -e events/event.json

# Navigate back to the original directory
cd -

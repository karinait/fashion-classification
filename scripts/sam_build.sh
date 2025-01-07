#!/bin/bash

# Navigate to the directory where the SAM template is located
cd ../fashion-classifier || exit

# Run the SAM build command
sam build --build-dir .aws-build

# Navigate back to the original directory
cd -

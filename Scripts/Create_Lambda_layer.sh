#!/bin/bash

# Variables
LAYER_NAME="my-lambda-layer" # Name of the Lambda layer
PYTHON_VERSION="python3.12"  # Python version to match your Lambda runtime
OUTPUT_DIR="lambda_layer"   # Directory to prepare the layer
ZIP_FILE="lambda_layer.zip" # Name of the zip file to create

# Create the necessary folder structure
echo "Creating folder structure..."
mkdir -p ${OUTPUT_DIR}/python/lib/${PYTHON_VERSION}/site-packages

# Install dependencies into the layer directory
echo "Installing dependencies from requirements.txt..."
pip install -r ../requirements.txt -t ${OUTPUT_DIR}/python/lib/${PYTHON_VERSION}/site-packages

# Zip the layer contents
echo "Packaging the layer..."
cd ${OUTPUT_DIR}
zip -r ../${ZIP_FILE} .
cd ..

echo "Lambda layer zip created successfully!"

# Publish the layer to AWS Lambda
#echo "Publishing the layer to AWS Lambda..."
#aws lambda publish-layer-version \
#    --layer-name ${LAYER_NAME} \
#    --zip-file fileb://${ZIP_FILE} \
#    --compatible-runtimes ${PYTHON_VERSION}

# Clean up
#echo "Cleaning up..."
#rm -rf ${OUTPUT_DIR} ${ZIP_FILE}

#echo "Lambda layer created and published successfully!"

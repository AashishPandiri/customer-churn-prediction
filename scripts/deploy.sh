#!/bin/bash

# This script is used for deploying the trained model to a production environment.

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
MODEL_DIR="path/to/model/directory"
DEPLOYMENT_DIR="path/to/deployment/directory"

# Copy the trained model to the deployment directory
echo "Deploying model..."
cp -r $MODEL_DIR/* $DEPLOYMENT_DIR/

# Restart the service (example for a Flask app)
echo "Restarting the application..."
# Replace 'your_flask_app' with the actual service name
sudo systemctl restart your_flask_app

echo "Deployment completed successfully!"
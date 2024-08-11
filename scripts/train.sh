#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the URLs for the datasets
DATASET_URL="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TEXT_URL="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

# Define the destination directory
DEST_DIR="../data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the Flickr8k dataset
echo "Downloading Flickr8k dataset..."
wget -O "$DEST_DIR/Flickr8k_Dataset.zip" "$DATASET_URL"

# Download the Flickr8k text files
echo "Downloading Flickr8k text files..."
wget -O "$DEST_DIR/Flickr8k_text.zip" "$TEXT_URL"

# Unzip the dataset
echo "Unzipping the Flickr8k dataset..."
unzip "$DEST_DIR/Flickr8k_Dataset.zip" -d "$DEST_DIR"

# Unzip the text files
echo "Unzipping the Flickr8k text files..."
unzip "$DEST_DIR/Flickr8k_text.zip" -d "$DEST_DIR"

# Clean up the zip files
echo "Cleaning up..."
rm "$DEST_DIR/Flickr8k_Dataset.zip"
rm "$DEST_DIR/Flickr8k_text.zip"

# Notify that the process is complete
echo "Flickr8k dataset and text files downloaded and saved to $DEST_DIR"

# Run the train.py Python script
echo "Running train.py..."
python3 train.py

# Notify that the training process is complete
echo "Training complete."

#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <source_directory> <destination_directory> <number_of_folders>"
    echo "Example: $0 /path/to/source /path/to/dest 100"
    echo ""
    echo "This script randomly selects and copies folders (directories) from source to destination."
    exit 1
}

# Check if required arguments are provided
if [ $# -ne 3 ]; then
    usage
fi

SOURCE_DIR="$1"
DEST_DIR="$2"
NUM_FOLDERS="$3"

# Validate source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Validate number of folders is a positive integer
if ! [[ "$NUM_FOLDERS" =~ ^[0-9]+$ ]] || [ "$NUM_FOLDERS" -le 0 ]; then
    echo "Error: Number of folders must be a positive integer."
    exit 1
fi

# Find all directories in source (excluding source itself)
echo "Scanning for folders in: $SOURCE_DIR"
TOTAL_FOLDERS=$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "Found $TOTAL_FOLDERS folders in source directory"

# Check if we have enough folders
if [ "$TOTAL_FOLDERS" -lt "$NUM_FOLDERS" ]; then
    echo "Warning: Only $TOTAL_FOLDERS folders available, but $NUM_FOLDERS requested."
    echo "Will copy all available folders."
    NUM_FOLDERS=$TOTAL_FOLDERS
fi

if [ "$TOTAL_FOLDERS" -eq 0 ]; then
    echo "No folders found in source directory."
    exit 1
fi

# Randomly select and copy folders
echo "Randomly selecting and copying $NUM_FOLDERS folders..."

count=0
find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | shuf -n "$NUM_FOLDERS" | while IFS= read -r folder; do
    count=$((count + 1))
    foldername=$(basename "$folder")
    echo "[$count/$NUM_FOLDERS] Copying folder: $foldername"

    # Copy the entire folder with all contents recursively
    cp -r "$folder" "$DEST_DIR/"

    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully copied: $foldername"
    else
        echo "  ✗ Failed to copy: $foldername"
    fi
done

echo "Folder copying completed!"

# Verification
COPIED_FOLDERS=$(find "$DEST_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Verification: $COPIED_FOLDERS folders now in destination directory"

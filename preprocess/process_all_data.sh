#!/bin/bash

# Shell script to process all ChemLLMBench data
# This script processes all data from /BioLatent/ChemLLMBench/data
# and outputs the processed data to /BioLatent/Bio-LatentCOT/data/ChemLLMBench
# with subdirectories for each task type

set -e  # Exit on any error

echo "Starting preprocessing of ChemLLMBench data..."

# Define input and output directories
INPUT_DIR="/BioLatent/ChemLLMBench/data"
OUTPUT_DIR="/BioLatent/Bio-LatentCOT/data/ChemLLMBench"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Run the preprocessing script
echo "Running preprocessing script..."
python /BioLatent/Bio-LatentCOT/code_train_sft/preprocess_chemllm.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    echo "Processed files are located in: $OUTPUT_DIR"

    # List the processed files with subdirectories
    echo "Directory structure:"
    find "$OUTPUT_DIR" -type f | sort
else
    echo "Error: Preprocessing failed!"
    exit 1
fi

echo "All data has been processed successfully."

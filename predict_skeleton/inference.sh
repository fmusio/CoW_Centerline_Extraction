#!/bin/bash

# Echo the Python path to verify environment
echo "Using Python from: $(which python)"

echo "dataset ID: $1"
echo "input folder: $2"
echo "output folder: $3"
echo "fold: $4"
echo "trainer: $5"
echo "config: $6"
echo "plan: $7"
echo "nnUNet_results: ${nnUNet_results}"

# Check if fold parameter is 'all' and set fold values accordingly
if [ "$4" = "all" ]; then
    fold="0 1 2 3 4"
    echo "Processing all folds: $fold"
else
    fold="$4"
    echo "Processing specified fold: $fold"
fi

# Run the nnUNetv2_predict command
nnUNetv2_predict -d $1 -i $2 -o $3 -f ${fold} -tr $5 -c $6 -p $7


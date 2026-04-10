#!/bin/sh

# Usage:
# ./mergecl.sh kernelSource.cl file1.cl file2.cl file3.cl

# Check if at least 2 arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 output_file input_file1 [input_file2 ...]"
    exit 1
fi

# First argument is the output file
output_file="$1"
shift

# Empty/create output file
: > "$output_file"

# Loop through remaining input files
for file in "$@"; do
    if [ -f "$file" ]; then
        cat "$file" >> "$output_file"
        echo "" >> "$output_file"   # Optional newline between files
    else
        echo "Warning: '$file' does not exist or is not a regular file."
    fi
done

echo "Merged files into '$output_file'"
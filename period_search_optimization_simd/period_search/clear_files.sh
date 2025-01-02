#!/bin/bash

# List of files to delete
files_to_delete=(
    "boinc_lockfile"
    "period_search_out"
    "period_search_state"
)

# Loop through each file and delete it if it exists
for file in "${files_to_delete[@]}"; do
    if [ -e "$file" ]; then
        rm "$file"
        echo "Deleted: $file"
    else
        echo "File not found: $file"
    fi
done

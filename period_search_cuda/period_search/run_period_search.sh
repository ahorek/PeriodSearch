#!/bin/bash

# Check if the executable name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <executable_filename>"
    exit 1
fi

# Path to the directory containing work units
work_units_dir="./work_units"

# Path to the executable inside the build subdirectory
executable="./build/$1"

# Check if the executable exists
if [ ! -f "$executable" ]; then
    echo "Error: Executable '$executable' not found in './build'."
    exit 1
fi

echo "Starting to process work units..."
echo "Listing files in $work_units_dir:"
ls $work_units_dir

# Find and iterate over each work unit in the directory
for work_unit in $(find "$work_units_dir" -type f -name 'input_*'); do
    echo "Checking file: $work_unit"

    # Ensure it is a file
    if [ -f "$work_unit" ]; then
        echo "Processing work unit: $work_unit"

        # Copy the current work unit to the build directory as "period_search_in"
        cp "$work_unit" "./build/period_search_in"
        
        echo "Copied $work_unit to ./build/period_search_in"
        
        # Change directory to ./build before running the executable
        pushd ./build > /dev/null
        echo "Changed directory to $(pwd)"
        
        # Run the executable
        echo "Running executable: $executable"
        "./$(basename $executable)"
        
        # Check if the executable ran successfully
        if [ $? -ne 0 ]; then
            echo "Error running the executable for work unit: $work_unit"
            exit 1
        fi
        
        # Wait for 2 seconds to ensure the executable finishes completely
        sleep 2

        # Change back to the original directory
        popd > /dev/null

        # Check if the "boinc_finish_called" file exists before proceeding
        if [ -f "./build/boinc_finish_called" ]; then
            echo "boinc_finish_called exists. Waiting for 1 second before proceeding."
            sleep 1
        else
            echo "boinc_finish_called does not exist."
        fi

        # Delete some files after processing the work unit
        if [ -f "./build/period_search_in" ]; then
            rm "./build/period_search_in"
            echo "Deleted period_search_in"
        fi
        
        if [ -f "./build/boinc_finish_called" ]; then
            rm "./build/boinc_finish_called"
            echo "Deleted boinc_finish_called"
        fi

        if [ -f "./build/period_search_state" ]; then
            rm "./build/period_search_state"
            echo "Deleted period_search_state"
        fi

        if [ -f "./build/boinc_lockfile" ]; then
            rm "./build/boinc_lockfile"
            echo "Deleted boinc_lockfile"
        fi

        # Extract the numbers from the input file name
        num=$(basename "$work_unit" | grep -oP '\d+_\d+')
        
        # Check if the output file exists and rename it
        if [ -f "./build/period_search_out" ]; then
            new_output_file="period_search_out_${num}_cuda"
            mv "./build/period_search_out" "./build/$new_output_file"
            
            # Move the renamed output file to the work_units directory
            mv "./build/$new_output_file" "$work_units_dir"
            
            echo "Processed work unit: $(basename "$work_unit"), output moved to $work_units_dir"
        else
            echo "Output file period_search_out not found for work unit: $(basename "$work_unit")"
        fi
    else
        echo "Work unit $work_unit is not a file."
    fi
done

echo "All done."

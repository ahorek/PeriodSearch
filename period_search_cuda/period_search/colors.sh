#!/bin/bash

# Print the 256 colors
for i in {0..255}; do
    printf "\033[38;5;${i}mColor ${i}\n"
done

# Reset to default color
printf "\033[0m"

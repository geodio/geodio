#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

# Clean previous builds
rm -rf build/ dist/ *.egg-info
rm -rf src/c_operands.c src/*.so src/*.pyd  # Remove only the Cython-generated .c file

# Run the Python setup to build everything using setuptools
python setup.py build_ext --inplace

# Clean up intermediate files
rm -rf build/ dist/ *.egg-info

# If the build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
else
    echo "Build failed."
    exit 1
fi

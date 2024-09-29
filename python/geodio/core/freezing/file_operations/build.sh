#!/bin/bash
# Compilation script for Linux, macOS, and Windows (with MinGW)

# Navigate to the script's directory (ensure you're in the right path)
cd "$(dirname "$0")"

# Clean up previous builds
rm -rf build
rm -rf src/file_operations.c src/*.so src/*.pyd  # Remove only the Cython-generated .c file

# Build the extension
python3 setup.py build_ext --inplace

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
else
    echo "Build failed."
    exit 1
fi

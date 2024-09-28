#!/bin/bash

# Stop the script if any command fails
set -e

# Define the build directory
BUILD_DIR="build"

# Check if the build directory exists; if not, create it
if [ ! -d "$BUILD_DIR" ]; then
  echo "Creating build directory..."
  mkdir $BUILD_DIR
fi

# Navigate to the build directory
cd $BUILD_DIR

# Run CMake to configure the project
echo "Configuring the project with CMake..."
cmake ..  # Adjust this if you need specific CMake options

# Run the build process with verbose output for easier debugging
echo "Building the project..."
cmake --build . --verbose

# Notify the user that the build has finished successfully
echo "Build completed successfully!"

# Optionally, list the generated files for convenience
echo "Generated files:"
ls -l

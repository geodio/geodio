#!/bin/bash

# Stop the script if any command fails
set -e

# Define the project root directory (relative to this script's location)
PROJECT_ROOT="$(dirname "$0")"

# Define the build directory at the root level
BUILD_DIR="${PROJECT_ROOT}/build"

# Define the source directory (where CMakeLists.txt is located)
SOURCE_DIR="${PROJECT_ROOT}/cpp"

# Define the Python bindings directory for geodio
PYTHON_BINDINGS_DIR="${BUILD_DIR}/python/geodio"

# Check if the build directory exists; if not, create it
if [ ! -d "$BUILD_DIR" ]; then
  echo "Creating build directory at root level..."
  mkdir -p "$BUILD_DIR"
fi

# Navigate to the build directory
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring the project with CMake..."
cmake "$SOURCE_DIR"  # Specify the source directory for out-of-source build

# Run the build process with verbose output for easier debugging
echo "Building the project..."
cmake --build . --verbose

# Notify the user that the build has finished successfully
echo "Build completed successfully!"

echo "Generated files in $PYTHON_BINDINGS_DIR:"
ls -l "$PYTHON_BINDINGS_DIR"
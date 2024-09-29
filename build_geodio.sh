#!/bin/bash

# Stop the script if any command fails
set -e

# Set project paths
PROJECT_ROOT="$(dirname "$0")"
CPP_DIR="${PROJECT_ROOT}/cpp"
BUILD_DIR="${PROJECT_ROOT}/build"
PYTHON_DIR="${PROJECT_ROOT}/python"

# Create the build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
  echo "Creating build directory at root level..."
  mkdir -p "$BUILD_DIR"
fi

# Navigate to the build directory
cd "$BUILD_DIR"

# Step 1: Run CMake to configure the C++ project
echo "Configuring the C++ project with CMake..."
cmake "$CPP_DIR"

# Step 2: Build the C++ project and bindings
echo "Building the C++ project and Python bindings..."
cmake --build . --verbose

# Step 3: Find the generated Python bindings file (with wildcard)
BINDING_FILE=$(find "${BUILD_DIR}/python/geodio/" -name "geodio_bindings*.so")

# Check if the binding file was found
if [ -z "$BINDING_FILE" ]; then
  echo "Error: Python bindings file not found!"
  exit 1
else
  echo "Python bindings successfully generated: ${BINDING_FILE}"
fi

# Step 4: Navigate to the Python directory for package setup
cd "$PYTHON_DIR"

# Step 5: Run the setup.py script to build and install the Python package with the custom binding file
echo "Running setup.py to install the geodio Python package with bindings..."
python3 setup.py install --binding-file="$BINDING_FILE"

# Step 6: Optional - Create a distribution (source and wheel)
echo "Creating a distribution package (source and wheel)..."
python3 setup.py sdist bdist_wheel --binding-file="$BINDING_FILE"

# Step 7: Optional - Display the contents of the 'dist' directory (distributable package)
echo "Distribution files generated in: ${PYTHON_DIR}/dist/"
ls -l "${PYTHON_DIR}/dist"

echo "Build and setup process completed successfully!"

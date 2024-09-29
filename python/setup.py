import os
import sys

from setuptools import setup, find_packages

# Default path to the compiled Python bindings
BINDING_FILE = os.path.join('geodio', 'geodio_bindings.so')

# Check if a custom binding file path is provided via command line arguments
for arg in sys.argv:
    if arg.startswith('--binding-file='):
        BINDING_FILE = arg.split('=')[1]
        sys.argv.remove(arg)  # Remove the argument so it doesn't interfere with setuptools

# Package configuration for geodio
setup(
    name='geodio',
    version='0.1',
    description='Geodio: High-performance multi-language bindings',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',  # Assuming README.md is in markdown format
    packages=find_packages(),  # Automatically find all packages in 'geodio/'
    package_dir={'geodio': 'geodio'},  # The main Python package is in the 'geodio' directory
    package_data={
        'geodio': [BINDING_FILE]  # Include the compiled shared library in the package
    },
    install_requires=[],  # Add any Python dependencies here
    zip_safe=False,
)
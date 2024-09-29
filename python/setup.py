from setuptools import setup, Extension
import os

# Define the C++ extension
module = Extension(
    'geodio.geodio_bindings',  # Python module name (geodio namespace)
    sources=['../cpp/src/bindings/geodio_binding.cpp'],  # Adjust paths as needed
    include_dirs=['../cpp/src'],  # Include C++ source directories
)

# Package configuration for geodio
setup(
    name='geodio',
    version='0.1',
    description='Geodio',
    ext_modules=[module],
    packages=['geodio'],  # Name of the Python package
    package_dir={'geodio': 'geodio'},  # Python modules are in the "geodio" directory
    install_requires=[],  # Add any required dependencies
    zip_safe=False,
)

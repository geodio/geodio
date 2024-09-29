import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the root directory as the directory containing this script
root_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the source files relative to the root directory
source_files = [
    os.path.join(root_dir, "src", "file_operations.pyx"),
    os.path.join(root_dir, "src", "FileOperations.c")
]

# Define the include directories
include_dirs = [
    np.get_include(),
    os.path.join(root_dir, "src")  # Ensure the C header file can be found
]

extensions = [
    Extension(
        name="file_operations",
        sources=source_files,
        include_dirs=include_dirs,
        extra_compile_args=["-std=c99", "-O3"]
    )
]

setup(
    name="file_operations",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Set the language level to Python 3
    ),
)

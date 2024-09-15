import os
import sys

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sysconfig

# Define the root directory as the directory containing this script
root_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the source files relative to the root directory
source_files = [
    os.path.join(root_dir, "src", "c_operands.pyx"),
    os.path.join(root_dir, "src", "LinearTransformation.c"),
    os.path.join(root_dir, "src", "Addition.c")
]

cuda_source_files = [
    os.path.join(root_dir, "src", "LinearTransformation.cu")
]

# Define the include directories
include_dirs = [
    np.get_include(),
    os.path.join(root_dir, "src")  # Ensure the C header file can be found
]

# Define library directories (you might need this depending on your system's setup)
library_dirs = [
    os.environ.get('LIBDIR', sysconfig.get_config_var('LIBDIR'))  # Use environment variable or fallback
]

# Optional: Add libraries you need, like crypt if needed
libraries = ['crypt']

# Extra Compile Arguments
extra_compile_args = ['-std=c99', '-O3']

# Extra Link Arguments
extra_link_args = []

# Add CUDA source files if GPU support is enabled
if "--use-cuda" in sys.argv:
    source_files.extend(cuda_source_files)
    library_dirs.append("cublas")

    extra_link_args.extend(['-lcudart', '-lcublas'])
    sys.argv.remove("--use-cuda")


extensions = [
    Extension(
        name="c_operands",
        sources=source_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="c_operands",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}  # Set the language level to Python 3
    ),
)

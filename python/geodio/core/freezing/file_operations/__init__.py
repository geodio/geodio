def compile_file_operations():
    import os
    import subprocess
    import sys
    # Set the root directory relative to this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Call the build script to compile the Cython extension
    subprocess.check_call([os.path.join(root_dir, "build.sh")])
    # Adjust sys.path to include the directory where the compiled module is located
    sys.path.insert(0,
                    os.path.join(root_dir, "core", "freezing",
                                 "file_operations"))


# compile_file_operations()

# Now import the module

import os
import subprocess

import numpy as np

from core.cell import Operand, Weight


def transpile_to_cpp(operand: Operand):
    cpp_code = operand.to_cpp()
    cpp_code_with_main = f"""
    #include <iostream>
    #include "functions.h"

    int main() {{
        auto result = {cpp_code};
        std::cout << "Result: " << result << std::endl;
        return 0;
    }}
    """
    with open("temp_code.cpp", "w") as cpp_file:
        cpp_file.write(cpp_code_with_main)

    subprocess.run(["g++", "temp_code.cpp", "-o", "temp_program"])
    subprocess.run(["./temp_program"])


# Example usage
if __name__ == "__main__":
    weight = Weight(np.array([1, 2, 3]))
    transpile_to_cpp(weight)

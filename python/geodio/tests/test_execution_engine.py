import numpy as np
from geodio import geodio_bindings

# Import operands from your Python implementation
from geodio.core.cell.operands import Add, Prod, Constant, Variable, \
    SigmoidActivation, LinearTransformation

# Helper function to convert AnyTensor to a Python Tensor
def tensor_from_anytensor(anytensor):
    return Tensor(anytensor)


def run_python_graph(graph):
    """
    Executes the Python graph using numpy operations.
    """
    return graph()


def run_cpp_graph(graph_wrapper, output_id):
    """
    Executes the graph using the C++ execution engine.
    """
    engine = geodio_bindings.ExecutionEngine()
    result = engine.forward(graph_wrapper.graph, output_id)
    return tensor_from_anytensor(result)


def compare_results(py_result, cpp_result):
    """
    Compare Python and C++ results numerically.
    Converts C++ result (AnyTensor) to numpy array if necessary.
    """
    py_data = py_result.data() if isinstance(py_result, Tensor) else py_result
    cpp_data = cpp_result.data() if isinstance(cpp_result,
                                               Tensor) else cpp_result

    # Convert to numpy arrays for comparison
    py_np = np.array(py_data)
    cpp_np = np.array(cpp_data)

    # Ensure that the arrays are numerically similar
    assert np.allclose(py_np, cpp_np,
                       atol=1e-6), f"Mismatch! Python: {py_np}, C++: {cpp_np}"


def test_graph_execution():
    """
    Comprehensive test that builds and compares Python and C++ graph execution results.
    """

    # Step 1: Create variables, constants, and weights
    x = Variable(0)
    W = Constant([[2.0, 1.0], [1.0, 2.0]])  # Weight matrix
    b = Constant([1.0, 2.0])  # Bias

    # Step 2: Create a simple computation graph in Python
    graph = Add(LinearTransformation(W, x),
                b)  # Linear transformation W * x + b
    graph = SigmoidActivation(graph)  # Apply sigmoid

    # Provide input for the variable x
    x_value = np.array([3.0, 5.0])

    # Step 3: Execute the Python graph
    py_result = run_python_graph(lambda: graph([x_value]))

    # Step 4: Create a corresponding C++ computational graph
    graph_wrapper = geodio_bindings.ComputationalGraph()

    # Add operands to the graph
    x_operand = geodio_bindings.Operand(geodio_bindings.OperandType.Variable,
                                        0, [])
    W_operand = geodio_bindings.Operand(geodio_bindings.OperandType.Constant,
                                        1, [])
    b_operand = geodio_bindings.Operand(geodio_bindings.OperandType.Constant,
                                        2, [])

    # Add constants to the C++ graph
    graph_wrapper.constants[1] = Tensor(
        [[2.0, 1.0], [1.0, 2.0]]).tensor  # W matrix
    graph_wrapper.constants[2] = Tensor([1.0, 2.0]).tensor  # Bias
    graph_wrapper.constants[3] = Tensor(x_value).tensor  # Input for x

    # Create a computation graph in C++
    graph_wrapper.operands[0] = x_operand
    graph_wrapper.operands[1] = W_operand
    graph_wrapper.operands[2] = b_operand
    linear_transformation = geodio_bindings.Operand(
        geodio_bindings.OperandType.LinearTransformation, 3, [0, 1])
    addition = geodio_bindings.Operand(geodio_bindings.OperandType.Add, 4,
                                       [3, 2])
    sigmoid = geodio_bindings.Operand(geodio_bindings.OperandType.Sigmoid, 5,
                                      [4])

    # Add the operands to the graph
    graph_wrapper.operands[3] = linear_transformation
    graph_wrapper.operands[4] = addition
    graph_wrapper.operands[5] = sigmoid

    # Step 5: Execute the C++ graph
    cpp_result = run_cpp_graph(graph_wrapper, 5)

    # Step 6: Compare Python and C++ results
    compare_results(py_result, cpp_result)

    print("Python and C++ execution results are numerically identical!")


if __name__ == "__main__":
    test_graph_execution()

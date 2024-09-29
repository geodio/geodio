from tensor_wrapper import Tensor

# Create tensors
t1 = Tensor([1.0, 2.0, 3.0], [3])
t2 = Tensor([4.0, 5.0, 6.0], [3])

# Perform element-wise addition
result = t1 + t2
print(result)  # Output: <Tensor shape=[3] data=[5.0, 7.0, 9.0]>

# Matrix multiplication
mat1 = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
mat2 = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
mat3 = mat1.matmul(mat2)
print(str(mat3))

# Summing a tensor
print(t1.sum([0]))  # Sum along axis 0


from core.cpp_wrappers.geodio_bindings import (ComputationalGraph, Operand, OperandType,
                             ExecutionEngine, initialize_operations)

graph = ComputationalGraph()
initialize_operations()
graph.add_operand(0, Operand(OperandType.Add, 0, [1, 2]))
graph.add_operand(1, Operand(OperandType.Constant, 1, []))
graph.add_operand(2, OperandType.Constant, [])

graph.add_constant(1, Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]).tensor)
graph.add_constant(2, Tensor([4.0, 3.0, 2.0, 1.0], [2, 2]).tensor)

print(graph.constants)  # This should show the full map
print(graph.constants[1])
result = ExecutionEngine.forward(graph, 0, [])
print(result)

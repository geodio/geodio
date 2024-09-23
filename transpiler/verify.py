from c_operands import LinearTransformationWrapper, AdditionWrapper

# Initialize linear transformation with dimensions (3, 2)
lt = LinearTransformationWrapper(3, 2)

# Set inputs, weights, and biases dynamically
lt.set_input([1.0, 2.0, 3.0])
lt.set_weights([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
lt.set_bias([0.01, 0.02])

# Perform the linear transformation
lt.linear_transform()
lt_result = lt.get_result()
print("Linear Transformation Result:", lt_result)

# Initialize addition with size 3
add = AdditionWrapper(3)

# Set operands dynamically
add.set_operands([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

# Perform the addition
add.add()
add_result = add.get_result()
print("Addition Result:", add_result)

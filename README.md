# Geodio Framework: A Readme Overview
## Introduction

The **Geodio** framework began as an exploration into **Gene Expression Programming (GEP)** for creating neural network architectures. Over time, it evolved into a powerful framework for designing, optimizing, and managing complex neural networks, where every part is an independent entity called an **Operand**.

### Core Idea

At its core, Geodio enables you to build neural networks by composing these Operands into complex, nested trees. These trees are capable of:

- **Genetic mutation**
- **Gradient descent optimization**
- **Backpropagation**
- **Reuse and optimization at the tree level**
- **Compilation and interpretation**

The goal is to simplify the creation and experimentation with new neural network architectures, allowing both flexibility and scalability.

## Geodio's Journey

Initially, Geodio was a Python-based framework focused on **Gene Expression Programming (GEP)**. Over time, it transitioned into a blend of **Derivable Programming** and GEP, supporting optimizations via backpropagation in Python.

However, due to Python's performance limitations, Geodio shifted to a **C++ implementation**. This new approach allows for:

- **Faster interpretation**
- **Efficient GPU optimization**

With the move to C++, a **transpiler** _will_ be introduced, enabling Geodio to bootstrap itself using its own framework, paving the way for further optimization and extensibility.

## Yaguar: The Language for Geodio

**Yaguar** was born out of the need to design Operands more easily. Python had limitations in how it handled the exact nature of Geodio’s operations, especially when dealing with gene expression, derivation, and backpropagation.

Yaguar addresses this by offering a flexible syntax designed specifically for Geodio’s needs. It simplifies the process of creating complex operands, managing recursive functions, and controlling dynamic execution flows through jumps, labels, and conditionals.

## Features of Geodio

Geodio is designed to be highly extensible, allowing you to:

- **Create recursive, nested computational graphs** that can return function pointers.
- **Pass and return functions as Operands**, making them first-class citizens.
- **Implement dynamic control flow** with jumps, labels, and conditional tokens.
- **Optimize entire computational trees or subgraphs** via backpropagation. **(not yet implemented)**
- **Extend with low-level control** through future support for inline assembly, C, or C++ code. **(not yet implemented)**

### Key Differences with TensorFlow

| Feature                  | Geodio Framework                                           | TensorFlow                                  |
|--------------------------|------------------------------------------------------------|---------------------------------------------|
| **Functions as Operands** | Functions can be passed, returned, and optimized.          | Wrapped functions, but not embedded in the graph. |
| **Subgraphs and Recursion** | Supports recursive, nested computational graphs.           | Static execution graphs with limited dynamic flow. |
| **Dynamic Control Flow**  | Full control with jump, label, and conditionals.           | Limited to `tf.while_loop` and `tf.cond`.   |
| **Function Optimization** | **(not yet implemented)** Functions are optimized via backpropagation. | No direct function optimization.            |
| **GPU Acceleration**      | **(not yet implemented)** Custom GPU kernels for functions and graphs. | GPU acceleration, but less flexible.        |

### What Geodio Can Achieve That TensorFlow Can’t

- **Recursive Neural Networks**: Geodio supports recursive architectures where functions can return subgraphs or other functions.
- **Self-Modifying Networks**: Networks can modify themselves based on inputs, allowing self-learning architectures.
- **Macros and Inline Code**: **(not yet implemented)** Geodio offers low-level control by allowing inline assembly or C/C++ code.
- **Desugaring Complex Objects**: **(not yet implemented)** Complex objects are broken down into simpler functional representations, providing a more functional programming approach.
- **Custom GPU Execution**: **(not yet implemented)** Geodio enables deep integration of custom CUDA/OpenCL kernels, offering more flexible GPU optimization.

---

## Examples

### Simple Neural Network Example

You can create and run a simple neural network using Geodio as follows:

```yaguar
l = Linear(2, 1) >> Sigmoid() >> Linear(1, 2) >> Sigmoid()
print(l([1, 2]))
```

This defines a neural network with two layers and applies the **Sigmoid** activation function.

### Sum Function Example

Recursive functions are a key feature in Yaguar, enabling you to write functions like summing numbers in a range:

```yaguar
yay sum(start, end):
    yay qwerty(current, total):
        current >= end ? total
        ?? qwerty(current + 1, current + total)
    qwerty(start, 0)

print(sum(1, 10))  !> Prints 45.0
```

### Recursive While Loop Example

Using `jmp` for loop-like functionality:

```yaguar
i = 0
sum = 0

while:
    i < 4 ?
        sum = sum + i ^ 2
        i = i + 1
        jmp while
print(sum)  !> Prints 144
```

---

## Contributing

Geodio is in its early stages of development, and there's a lot of room for growth. Contributions are highly welcome! However, due to the project's complexity and infancy, contributors are encouraged to [email me](mailto:polenciucrares%40gmail.com?subject=Geodio%20Contributing) to discuss potential contributions and the project's direction.

---

## Future Direction

The vision for Geodio is to provide a framework capable of **self-building** neural networks, pushing towards **Artificial General Intelligence (AGI)**. While the current development focuses on optimization and architecture experimentation, the ultimate goal is to create a system that can autonomously evolve its own intelligence.

---

For more examples and usage, see the `/examples` folder.
**THIS README IS WORK IN PROGRESS**

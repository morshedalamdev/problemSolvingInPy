import numpy as np
# Basic NumPy Operations
arr1 = np.array([1, 2, 3, 4, 5])
print("1D Array:\n", arr1)
print("--------------------------------\n")

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr2)
print("--------------------------------\n")
# Outputs
print("Array Size:", arr2.size)
print("--------------------------------")
print("Array Shape:", arr2.shape)
print("--------------------------------")
print("Array Dimensions:", arr2.ndim)
print("--------------------------------")
print("Array Data Type:", arr2.dtype)
print("--------------------------------\n")


# Special Matrix Arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)
random = np.random.rand(2, 3)
# Outputs
print("Zeros matrix:\n", zeros)
print("--------------------------------")
print("Ones matrix:\n", ones)
print("--------------------------------")
print("Identity matrix:\n", identity)
print("--------------------------------")
print("Random matrix:\n", random)
print("--------------------------------\n")


# Mathematical Operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# Outputs
print("Addition: ", a + b)
print("--------------------------------")
print("Subtraction: ", a - b)
print("--------------------------------")
print("Multiplication: ", a * b)
print("--------------------------------")
print("Division: ", a / b)
print("--------------------------------")
print("Dot product: ", np.dot(a, b))
print("--------------------------------\n")


# Indexing and Slicing
arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nElement at (1,2):", arr3[1, 2])
print("--------------------------------")
print("First row:", arr3[0, :])
print("--------------------------------")
print("Second column:", arr3[:, 1])
print("--------------------------------\n")


arr4 = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
print("Original Array:\n", arr4)
print("--------------------------------")
print("Neighbors of Node 0:", np.where(arr4[0] == 1)[0])

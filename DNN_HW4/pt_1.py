# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jun-hyeok/SUP5001-41_Deep-Neural-Networks_2022Spring/blob/main/DNN_HW4/pt_1.ipynb)

# %% [markdown]
# # DNN HW4 : Part 1
#
# 2022.03.16
# 박준혁

# %%
from util import *

# %% [markdown]
# ## Part 1: Python

# %% [markdown]
# ### 1-1. Python: List

# %%
print("1-1", color=GRAY)

# define
list1 = [1, 2, 3]

# index
print(f"{type(list1) = }")
print(f"{list1[0], list1[-1] = }")

# element
list1[1] = "str1"
print(f"{list1[1] = }")

# range
list2 = range(10)
print(f"{type(list2) = }")

# type cast
list2 = list(list2)
print(f"{type(list2) = }")

# %% [markdown]
# ### 1-2. Python: List

# %%
print("1-2", color=GRAY)

# slice range
print(f"{list2[2:4] = }")

# slice from # to end
print(f"{list2[2:] = }")

# slice from start to #
print(f"{list2[:2] = }")

# slice all (from start to end)
print(f"{list2[:] = }")

# slice step
print(f"{list2[::2] = }")

# slice neg index
print(f"{list2[:-1] = }")

# assign new list to sliced list
print()
print(f"{list2 = } : before")
list2[2:4] = [8, 9]
print("list2[2:4] = [8, 9]", color=GRAY, input=True)
print(f"{list2 = } : after")

print()
print(f"{list2 = } : before")
list2[2:5] = [8, 9]
print("list2[2:4] = [8, 9]", color=GRAY, input=True)
print(f"{list2 = } : after")

# %% [markdown]
# ### 1-3. Python: For Loop with List

# %%
print("1-3", color=GRAY)

# define animals list
animals = ["cat", "dog", "monkey"]

# print for loop
for animal in animals:
    print(f"{animal = }")
print()

# define nums list
nums = [0, 1, 2, 3, 4]
# empty list
squares = []

# calculate square of nums
for num in nums:
    squares.append(num**2)
print(f"{squares = }")

# %% [markdown]
# ### 1-4. Python: Function

# %%
print("1-4", color=GRAY)

# define function sign(x)
def sign(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"


# print sign of [-1, 0, 1]
for x in [-1, 0, 1]:
    print(f"{x = :2}, {sign(x) = }")

# %% [markdown]
# ### 1-5. Python: Function

# %%
print("1-5", color=GRAY)

# define function hello(name, loud=False)
def hello(name, loud=False):
    if loud:
        print(f"HELLO, {name.upper()}!")
    else:
        print(f"Hello, {name}!")


# hello loud default
print("hello('Bob')", color=GRAY, input=True)
hello("Bob")

# hello loud
print("hello('Fred', loud=True)", color=GRAY, input=True)
hello("Fred", loud=True)

# %% [markdown]
# ### 1-6. Python: zip

# %%
print("1-6", color=GRAY)

# print 1 4 7  / 2 5 8 / 3 6 9
for x, y, z in zip(range(1, 4), range(4, 7), range(7, 10)):
    print(f"{x, y, z = }")
print()

# error
try:
    for x, y in zip(range(1, 4), range(4, 7), range(7, 10)):
        print(f"{x, y, z = }")
except Exception as e:
    print(f"x, y, z = {type(e).__name__}: {e}", color=RED)

# %% [markdown]
# ### 1-7. Python: Magic method \_\_init\_\_()

# %%
print("1-7", color=GRAY)

# define class HelloWorld
class HelloWorld:
    def __init__(self):
        print("init")


print("helloworld = HelloWorld()", color=GRAY, input=True)
helloworld = HelloWorld()

# %% [markdown]
# ### 1-8. Python: Magic method \_\_call\_\_()

# %%
print("1-8", color=GRAY)


class HelloWorld:
    def __init__(self):
        print("init", color=GRAY)


print("helloworld = HelloWorld()", color=GRAY, input=True)
helloworld = HelloWorld()

# helloworld is callable?
print(f"{callable(helloworld) = }")
try:
    print(f"{helloworld() = }")
except Exception as e:
    print(f"helloworld() = {type(e).__name__}: {e}", color=RED)
print()


class HelloWorld:
    def __init__(self):
        print("init", color=GRAY)

    def __call__(self):
        print("Hello world")


print("helloworld = HelloWorld()", color=GRAY, input=True)
helloworld = HelloWorld()

print("helloworld()", color=GRAY, input=True)
helloworld()

# %% [markdown]
# ## Part 1: NumPy

# %% [markdown]
# ### NumPy: Install

# %%
try:
    import numpy as np
except ImportError:
    print("numpy is not installed", color=RED)
    import pip

    pip.main(["install", "numpy"])
else:
    print("numpy is installed", color=GREEN)

# %% [markdown]
# ### 1-9. Numpy: np.array

# %%
print("1-9", color=GRAY)

import numpy as np

# rank 1 array
a = np.array([1, 2, 3])

print(f"{type(a) = }")
print(f"{a.shape = }")
print(f"{a.ndim = }")
print(f"{a[0], a[1], a[2] = }")
print()

print(f"{a = } : before")
a[0] = 5
print("a[0] = 5", color=GRAY, input=True)
print(f"{a = } : after")
print()

# rank 2 array
b = np.array([[1, 2, 3], [4, 5, 6]])


print(f"{b.shape = }")
print(f"{b.ndim = }")
print(f"{b[0, 0], b[0, 1], b[0, 2] = }")

# %% [markdown]
# ### 1-10. Numpy: np.zeros, np.ones, np.full, np.eye, np.random

# %%
print("1-10", color=GRAY)

import numpy as np

# array with all zeros
a = np.zeros((2, 2))
print(f"{a = }")

# array with all ones
b = np.ones((1, 2))
print(f"{b = }")

# array with the specific value
c = np.full((2, 2), 7)
print(f"{c = }")

# 2x2 identity matrix
d = np.eye(2)
print(f"{d = }")

# array with random values
e = np.random.random((2, 2))
print(f"{e = }")

# %% [markdown]
# ### 1-11. Numpy: np.where

# %%
print("1-11", color=GRAY)

import numpy as np

# array with 0 to 9
a = np.arange(10)

# np.where(condition)
print(f"{np.where(a < 5) = }")

# np.where(condition, x, y)
print(f"{np.where(a < 5, a, 10 * a) = }")

# np.where(condition, x, y)
print(f"{np.where(a < 4, a, -1) = }")

# %% [markdown]
# ### 1-12. Numpy: Array slice

# %%
print("1-12", color=GRAY)

import numpy as np

# 3x4 array
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# slice the array a, the first 2 rows and column 1, 2
# shape of sub-array is (2, 2)
b = a[:2, 1:3]

# sub-array refer to the same array as original array, so
# change the sub-array will change the original array as well
# b[0, 0] == a[0, 1]
print(f"{b[0,0] == a[0,1] = }", color=GRAY)
print(f"{a[0, 1] = } : before")
b[0, 0] = 77
print("b[0, 0] = 77", color=GRAY, input=True)
print(f"{a[0, 1] = } : after")

# %% [markdown]
# ### 1-13. Numpy: Array reshape

# %%
print("1-13", color=GRAY)

import numpy as np

# 3x4 array
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# reshape the array a to be a 2x6 array
b = np.reshape(a, (2, 6))
print(f"{b = }")

# reshape also refer to the same array as original array, so
# change the reshaped-array will change the original array as well
# b[0, 0] == a[0, 0]
print(f"{b[0,0] == a[0,0] = }", color=GRAY)
print(f"{a[0, 0] = } : before")
b[0, 0] = 77
print("b[0, 0] = 77", color=GRAY, input=True)
print(f"{a[0, 0] = } : after")

# %% [markdown]
# ### 1-14. Numpy: Array reshape

# %%
print("1-14", color=GRAY)

import numpy as np

# 2 arrays with shape (2, 2)
a = np.array([[1, 0], [0, 1]])
print(f"{a = }", color=GRAY, input=True)
b = np.array([[4, 1], [2, 2]])
print(f"{b = }", color=GRAY, input=True)

# dot product of a and b
print(f"{np.dot(a, b) = }")

# %% [markdown]
# ## Part 1: Pytorch

# %% [markdown]
# ### Pytorch: Install

# %%
try:
    import torch
except ImportError:
    print("torch is not installed", color=RED)
    import pip

    pip.main(["install", "torch"])
else:
    print("torch is installed", color=GREEN)

# %% [markdown]
# ### 1-15. Pytorch: Tensor

# %%
print("1-15", color=GRAY)

import numpy as np
import torch

# float type tensor
t1 = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
t2 = torch.tensor(np.arange(7))

print(f"{t1.shape = }")
print(f"{t2.shape = }")
print(f"{t1.dim() = }")
print(f"{t1.size() = }")
print(f"{t1[:2] = }")
print(f"{t1[3:] = }")

# %% [markdown]
# ### 1-16. Pytorch: NumPy array vs PyTorch tensor

# %%
print("1-16", color=GRAY)

import numpy as np
import torch

b = np.arange(7)
t1 = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
print(f"{b = }")
print(f"{t1 = }")
print(f"{type(b) = }")
print(f"{type(t1) = }")
print()

# transform numpy array to torch tensor
tt = torch.tensor(b)
t_from = torch.from_numpy(b)
print(f"{tt = }")
print(f"{t_from = }")
print(f"{type(tt) = }")
print(f"{type(t_from) = }")

# %% [markdown]
# ### 1-17. Pytorch: NumPy array vs PyTorch tensor

# %%
print("1-17", color=GRAY)

print(f"{tt = } : before")
print(f"{t_from = } : before")
b[0] = -10
print("b[0]= -10", color=GRAY, input=True)
print(f"{tt = } : after")
print(f"{t_from = }: after")
print()

# transform torch tensor to numpy array
t_to_np = t_from.numpy()
print(f"{t_to_np = }")
print(f"{type(t_to_np) = }")

# %% [markdown]
# ### 1-18. Pytorch: Broadcasting

# %%
print("1-18", color=GRAY)

import numpy as np
import torch

# addition of two tensors
m1 = torch.FloatTensor([[3, 3]])
print(f"{m1 = }", color=GRAY, input=True)
m2 = torch.FloatTensor([[2, 2]])
print(f"{m2 = }", color=GRAY, input=True)
print(f"{m1 + m2 = }")

# vector + scalar
m1 = torch.FloatTensor([[1, 2]])
print(f"{m1 = }", color=GRAY, input=True)
m2 = torch.FloatTensor([3])
print(f"{m2 = }", color=GRAY, input=True)
print(f"{m1 + m2 = }")

# 2x1 vector + 1x2 vector
m1 = torch.FloatTensor([[1, 2]])
print(f"{m1 = }", color=GRAY, input=True)
m2 = torch.FloatTensor([[3], [4]])
print(f"{m2 = }", color=GRAY, input=True)
print(f"{m1 + m2 = }")

# %% [markdown]
# ### 1-19. Pytorch: torch.mul vs torch.matmul

# %%
print("1-19", color=GRAY)

m1 = torch.FloatTensor([[1, 2], [3, 4]])
print(f"{m1 = }", color=GRAY, input=True)
m2 = torch.FloatTensor([[1], [2]])
print(f"{m2 = }", color=GRAY, input=True)
print(f"{m1 * m2 = }")
print(f"{m1.mul(m2) = }")
print(f"{m1.matmul(m2) = }")

# %% [markdown]
# ### 1-20. Pytorch: torch.view (np.reshape in PyTorch)

# %%
print("1-20", color=GRAY)

import numpy as np
import torch

t = np.arange(12).reshape(-1, 2, 3)
floatT = torch.FloatTensor(t)

print(f"{floatT.shape = }")
print(f"{floatT.view([-1, 3]) = }")

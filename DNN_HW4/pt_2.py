# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jun-hyeok/SUP5001-41_Deep-Neural-Networks_2022Spring/blob/main/DNN_HW4/pt_2.ipynb)

# %% [markdown]
# # DNN HW4 : Part 2
#
# 2022.03.16
# 박준혁

# %%
import numpy as np
import torch

from util import *

# %% [markdown]
# ## Part 2: Exercise

# %% [markdown]
# ##### 2-1. Use numpy to generate an array with values from 10 to 49
# (ex. 10, 11, ... 49)

# %%
print("2-1", color=GRAY)

# numpy array with 10 to 49
a = np.arange(10, 50)
print(f"{a = }")

# %% [markdown]
# ##### 2-2.  Use numpy to generate an array with values from 49 to 10
# (ex. 49, 48, ... 10)

# %%
print("2-2", color=GRAY)

# numpy array with 49 to 10
a = np.arange(49, 10, -1)
print(f"{a = }")

# %% [markdown]
# ##### 2-3 Find an array of indices with the same elements in a and b
# (Hint: np.where)

# %%
print("2-3", color=GRAY)

# input
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
print(f"{a = }", color=GRAY, input=True)
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
print(f"{b = }", color=GRAY, input=True)

# output: find an array of indices with the same elements in a and b.
print(f"{np.where(a == b)[0] = }")

# %% [markdown]
# ##### 2-4. Use broadcasting to find the result of adding 1 to a

# %%
print("2-4", color=GRAY)

# input
a = np.arange(9).reshape(3, 3)

# output: a + 1 using broadcasting
print(f"{a + 1 = }")

# %% [markdown]
# ##### 2-5. Change the first and second rows of a

# %%
print("2-5", color=GRAY)

# input
a = np.arange(9).reshape(3, 3)

# output: change the first and second rows of a
print(f"{a[[1, 0, *range(a.shape[0])[2:]], :] = }")

# %% [markdown]
# ##### 2-6. Find a for the following output

# %%
print("2-6", color=GRAY)

# output
print("a.shape = (2, 3, 4)", color=GRAY)
print(f"a = {np.arange(24).reshape(2, 3, 4)}", color=GRAY)

# input
a = input("a = ")
print(f"{a = !s}", input=True)
a = eval(a)

print(f"{a.shape = }")
print(f"{a = }")

# %% [markdown]
# ##### 2-7. Slice a for the following output

# %%
print("2-7", color=GRAY)

# output
print(f"{a[0, 1:, 2:]}", color=GRAY)

# input
idx = input("a[???] << ")
print(f"a[{idx}]", input=True)
print(eval(f"a[{idx}]"))

# %% [markdown]
# ##### 2-8. Use torch.view to get the following output

# %%
print("2-8", color=GRAY)

import torch
import numpy as np

t = np.arange(24)
floatT = torch.FloatTensor(t)

# output
print(f"{t.shape = }", color=GRAY)
print(f"{floatT.view([-1, 6])[2:, (1, 4)]}", color=GRAY)

# input
arg = input("floatT.view([???])[] << ")
idx = input("floatT.view([])[???] << ")
print(f"floatT.view([{arg}])[{idx}]", input=True)
print(eval(f"floatT.view([{arg}])[{idx}]"))

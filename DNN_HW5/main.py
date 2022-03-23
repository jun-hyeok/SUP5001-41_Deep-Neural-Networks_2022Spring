# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jun-hyeok/SUP5001-41_Deep-Neural-Networks_2022Spring/blob/main/DNN_HW5/main.ipynb)

# %% [markdown]
# # DNN HW5 : #9
#
# 2022.03.23
# 박준혁

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %% [markdown]
# Create XOR dataset with torch.FloatTensor

# %%
# xor dataset
x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = torch.FloatTensor([[0], [1], [1], [0]])

# %% [markdown]
# 1. NN model - 10 hidden layer with 4 nodes each

# %%
# neural network 10 hidden layers with 4 nodes each
class NN10(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc5 = nn.Linear(4, 4)
        self.fc6 = nn.Linear(4, 4)
        self.fc7 = nn.Linear(4, 4)
        self.fc8 = nn.Linear(4, 4)
        self.fc9 = nn.Linear(4, 4)
        self.fc10 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.sigmoid(self.fc10(x))
        return x


# %%
nn10 = NN10()
optimizer10 = optim.SGD(nn10.parameters(), lr=0.1)
epochs = 10000
for epoch in range(epochs):
    optimizer10.zero_grad()
    y_pred10 = nn10(x)
    ce10 = F.binary_cross_entropy(y_pred10, y)
    ce10.backward()
    optimizer10.step()
    if epoch % 1000 == 0:
        print("Epoch: {:4d}/{}".format(epoch, epochs), end=" ")
        print("Cost: {:.6f}".format(ce10.item()))

# %% [markdown]
# 2. NN model - 2 hidden layer with 4 nodes each

# %%
# neural network 2 hidden layers with 4 nodes each
class NN02(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


# %%
nn02 = NN02()
optimizer02 = optim.SGD(nn02.parameters(), lr=0.1)
epochs = 10000
for epoch in range(epochs):
    optimizer02.zero_grad()
    y_pred02 = nn02(x)
    ce02 = F.binary_cross_entropy(y_pred02, y)
    ce02.backward()
    optimizer02.step()
    if epoch % 1000 == 0:
        print("Epoch: {:4d}/{}".format(epoch, epochs), end=" ")
        print("Cost: {:.6f}".format(ce02.item()))

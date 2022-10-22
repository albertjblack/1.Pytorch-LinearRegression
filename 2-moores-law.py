# %% Imports
import torch
from torch import nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
# %% Data (Exponential growth)
X = [] # N by D
Y = [] # N by K

with open("./data/moore.csv", "r") as file:
  lines = file.readlines(); 
  lines = [x.strip('\n').split('\t') for x in lines]

pattern = re.compile(r'[^\d]+') 
for line in lines:
  x = int(pattern.sub('',line[2].split('[')[0]))
  y = int(pattern.sub('',line[1].split('[')[0]))
  X.append(x)
  Y.append(y)

X = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1,1)
plt.scatter(X,Y); 

# %% Get Linearity
Y = np.log(Y)
plt.scatter(X,Y)

# %% Preprocessing (Standardize/Normalize x and y) || Keep mean, std for later use || Later on we'll have to reverse the transformation to go back to the original units || float32 always in torch
Y = np.log(Y) # get it linearly
X_mean = X.mean()
X_std = X.std()
Y_mean = Y.mean()
Y_std = Y.std()
X = (X-X_mean)/X_std
X = X.astype(np.float32)
Y = (Y-Y_mean)/Y_std
Y = Y.astype(np.float32)
plt.scatter(X,Y)

# %% Modeling
model = nn.Linear(1,1)
learning_rate = 0.1
momentum = 0.7

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)

features = torch.from_numpy(X) # to tensors
labels = torch.from_numpy(Y)

# %% Main training loop
n_epochs = 100
losses = []
for it in range(n_epochs):
  optimizer.zero_grad()

  preds = model(features)
  loss = criterion(preds,labels)
  losses.append(loss.item())

  loss.backward()
  optimizer.step()

  print(f"Epoch: {it+1}/{n_epochs} | Loss: {loss.item():.4f}")

# %% Plot losses
plt.plot(losses)

# %% Line of best fit
preds_detached = preds.detach().numpy()
plt.plot(X,Y,"ro",label="Data",)
plt.plot(X,preds_detached)
plt.legend()
plt.show()

# %% Model weights
# since we took log. a = w(sy/sx) -> y int
w = model.weight.data.numpy() 
b = model.bias.data.numpy()
a = w[0][0] * (Y_std/X_std)
# taking a into exponential r = e^a
r = 2.71 ** a
# how long it takes for c to double 
t_to_double = np.log(2)/a
# %%

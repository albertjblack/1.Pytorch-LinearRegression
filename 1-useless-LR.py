# %% importing
import torch
import numpy as np
import pandas as pd
from torch import nn

# %% creating the model
"""
Pytorch does not work with numpy arrays
X, Y are numpy arrays of the data loaded
- numpy does float64 and torch floa32
inputs,target are not X,Y
"""
model = nn.Linear(1,1); 

# %% data
"""
We want our data to be NumSamples * NumDimensions
"""
N = 4; 
X = np.arange(start=1, stop=5);
Y = np.arange(start=1, stop=4);
X = X.rehape(N,1); 
Y = Y.rehape(N,1); 
inputs = torch.from_numpy(X.astype(np.float32)); 
labels = torch.from_numpy(Y.astype(np.float32)); 



# %% Train the model pt.1 (Loss & Optimizer)
"""
Define loss and optimizer
- criterion is object of MSELoss
- optimizer -> simplest is SGD (model.parameters(), learning_rate=0.1)
"""
criterion = nn.MSELoss(); 
optimizer = torch.optim.SGD(model.parameters(),lr=0.1); 

# %% Train the model pt.2 (Gradient Descent Loop)
"""
epoch -> each iteration of a loop
model(inputs) -> preds
pytorch allows us to use objects as functions
"""
n_epochs = 30; 
for it in range(n_epochs):
  # zero the parameter gradients || boilerplate -> pytorch accumulates gradient with loss.backward()
  optimizer.zero_grad(); 

  # forward pass
  preds = model(inputs); 
  loss = criterion(preds, labels); 

  loss.backward(); # backward || calculating the gradients dl/dm and dl/db 
  optimizer.step(); # update weights

# %% predictions
"""
Convert preds back to a numpy array
- normally you would use the numpy array funcition 
- pytorch does this thing behind the scenes where it creates a graph and keeps track of the gradients in the graph
"""
# detach tensor from the graph and convert it back to a numpy array
predictions = preds.detach().numpy() 
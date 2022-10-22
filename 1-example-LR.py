# %% Linear Regression Line of Best Fit
import imp
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# %% Data-Set
N = 20 # Generate 20 Data Points || N = samples, D = features
X = np.random.random(N) * 10 - 5;  # (-5, 5)
Y = 0.5 * X - 1 + np.random.randn(N);  # a line plus some noise
plt.scatter(X,Y); 
"""
- our true values that our model is going to try and find
m = 0.5
b = -1
"""

# %% PyTorch modeling
model = nn.Linear(1,1); 
learning_rate = 0.1; 

# %% PyTorch modeling 2
criterion = nn.MSELoss(); 
optimizer = torch.optim.SGD(model.parameters(), learning_rate); 

# %% Transforming X and Y. Set labels and labels (torch tensors)
X = X.reshape(N,1); 
Y = Y.reshape(N,1); 
labels = torch.from_numpy(X.astype(np.float32)); 
labels = torch.from_numpy(Y.astype(np.float32)); 

# %% Main training loop
"""
Gradient accumulation is extremely useful when working with large images/volumetric data, 
using low-end hardware, or training on multiple GPUs -> use larger batches without memory exhaustion.
'
' sometimes forced to use small batches during training -> slower convergence and lower accuracy.
' gradient accum: use a small batches but save gradients and update weights once every couple of batches
' ? Gradient accumulation: last step changes -> instead of updating netw on every batch, we store gradients
' and the weight update is done after several batches have been processed by the model.
' ! helps to imitate larger batches. If we wanted to use 32 images in one batch, but we crash @ 8
- -> we use batches of '8' images and update weights every '4'  // perform training in a slow machine
' 1) acum_iter 'in how many batches we would like to update weights'
' 2) condition weight update on the index of the running batch 
' 3) divide the running loss by acum_iter -> normalize the loss to reduce contribution of each mini batch
' this depends on how you compute the loss -> if you avg the loss within each batch -> no need to normalize
'
' https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html
"""
n_epochs = 30; 
losses = []; 
for it in range(n_epochs):
  optimizer.zero_grad();  # zero the gradients due to pytorch accumulation

  preds = model(labels);  # torch tensor
  loss = criterion(preds, labels)

  losses.append(loss.item()) # plot loss || .item() for tensors of single numbers
  loss.backward(); 

  optimizer.step(); # 1 step of gradient descent
  print(f"Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}"); 
# %% plot loss
plt.plot(losses);

# %% Make Predictions and Plot
"""
If you do not call detach: cannot convert into a numpy, so we can have another way of predicting with gradients 
"""
preds_detached = preds.detach().numpy(); 
plt.scatter(X,Y,label="Origin Data"); 
plt.plot(X,preds_detached,label="Fitted Line"); 
plt.legend(); 
plt.plot(); 

# %% Make predictions with gradients
"""
With the context manager we can instruct pytorch not to compute gradients to make the call inside the block
- out is not the linear regression line of best fit, but each individual prediction based on its label
"""
with torch.no_grad(): out = model(labels).numpy(); 
plt.scatter(X,Y, label="Training Data"); 
plt.scatter(X,out,label="Individual Prediction"); 
plt.plot(X, preds_detached, label="Best Fit Line"); 
plt.plot(); 

# %% Inspect the parameters of the model (m,b) to see if they are close to the real values || weigth & bias
w = model.weight.data.numpy(); # multi dimensional arr in case of other applications
b = model.bias.data.numpy(); 
print(w[0][0],b[0]); 
print(0.5,-1)

"""
Synthetic Data:
- Test the model by finding coded patern in synthetic data
- If it can find it :)
-- Else understand what they can and cannot do
"""
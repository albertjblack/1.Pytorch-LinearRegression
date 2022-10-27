# %% 0 - Imports
from statistics import mode
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# %% 1 - Data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer() # keys -> [data,target,target_names,DESCR,feature_names,filename]
X, Y = data.data, data.target # data.data.shape -> (569,30) 569 samples and 30 input features # targets -> 1d arr of 0s and 1s (569,1)

# a. splitting into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
N,D = X_train.shape

# b. normalizing || because feature ranges value can ve diff e.g 1m and 100 rg
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# c. converting data into torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1,1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1,1))

# %% 2 - Training
model = nn.Sequential(
  nn.Linear(D,1),
  nn.Sigmoid()
)
# a. modeling (NxD -> size of data)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# b. GD
n_epochs = 1000
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs) # whether or not is overfitting, if incr++
train_accs = np.zeros(n_epochs)
test_accs = np.zeros(n_epochs)
for it in range(n_epochs):
  # forward pass
  outputs = model(X_train) # model(features)
  loss = criterion(outputs,y_train)

  # backward pass
  loss.backward()
  optimizer.step()
  train_losses[it] = loss.item()


  # test_loss
  outputs_tests = model(X_test)
  test_losses[it] = criterion(outputs_tests,y_test).item() 

  # setting accuracies per iter
  train_accs[it] = np.mean(y_train.numpy()==np.round(model(X_train).detach().numpy()))
  test_accs[it] = np.mean(y_test.numpy()==np.round(model(X_test).detach().numpy()))

  # zeroing gradients
  optimizer.zero_grad()

  if (it+1)%50==0:  print(f"Epoch {it+1}\{n_epochs}, Train loss: {train_losses[it]}, Test loss: {test_losses[it]}")


# %% 3 - Evaluation

# a. plotting loss & acc per iteration // if losses are too diff -> overfitting
plt.plot(train_losses,label="train_losses")
plt.plot(test_losses,label="test_losses")
plt.legend()
plt.show()

plt.plot(train_accs,label="train_acc")
plt.plot(test_accs,label="test_acc")
plt.legend()
plt.show()

# b. model's accuracy
with torch.no_grad():
  p_train = np.round(model(X_train).numpy())
  train_acc = np.mean(y_train.numpy()==p_train)

  p_test = np.round(model(X_test).numpy())
  test_acc = np.mean(y_test.numpy()==p_test)
  print(f"Trn-Acc.:{train_acc:.4f} ({train_accs[n_epochs-1]}) | Tst-Acc.:{test_acc:.4f} ({test_accs[n_epochs-1]})")

"""
Accuracy here is the number right / total num of preds
- 1. get train preds by passing X_train to model which gives us p_train
  - 1.2. call np.round on p_train becasue these are just probabilities to 0,1
- 2. once we have our predictions we calc acc. by doing a pt wise comparison and taking the mean (1s/all) -> repeat process for test preds
"""

# %% Save and load model // model.state_dict -> orderedict -> weight, tensor, bias

# a. save
torch.save(model.state_dict(),"mymodel.pt")

# b. import to check if it saved correctly // process becomes easier when we do model classes -> 1 line of code

# recreate model the same way it was created
loaded_model = nn.Sequential(nn.Linear(D,1), nn.Sigmoid())
loaded_model.load_state_dict(torch.load("mymodel.pt"))

# c. evaluate the loaded model accs -> same results
with torch.no_grad():
  p_train = np.round(loaded_model(X_train).numpy())
  train_acc = np.mean(y_train.numpy()==p_train)
  p_test = np.round(loaded_model(X_test).numpy())
  test_acc = np.mean(y_test.numpy()==p_test)
  print(f"Train acc: {train_acc} || Test acc: {test_acc}")

"""
Download model from google colab -> dowanload from colab files or do this:
  from  google.colab import files
  files.download("mymodel.pt")
"""
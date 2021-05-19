import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from clearml import Task
task = Task.init(project_name="sdae", task_name="train")

BATCH = 5
EPOCH = 50
LR=0.001
RUN = "211390805"

data_path = os.path.join("data", RUN)

train = Tensor(pd.read_pickle(os.path.join(data_path, "train")).values).cuda()
test = Tensor(pd.read_pickle(os.path.join(data_path, "test")).values).cuda()
train_ds = TensorDataset(train, train)
train_dl = DataLoader(train_ds, batch_size=BATCH)

class AE(nn.Module):
    def __init__(self, input_size, num_layers=3, latent=[2000,1000,500]):
        super().__init__()
        assert num_layers == len(latent)
        self.input = input_size
        self.fce = nn.Linear(self.input, latent[0])
        self.fcd = nn.Linear(latent[0], self.input)

    def forward(self, x):
        #for layer in 
        x = self.fce(x)
        x = self.fcd(x)
        return x

model = AE(input_size=train.shape[1]).cuda()
loss_func = F.mse_loss
optm=optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    for xb, yb in train_dl:

        pred = model(xb)
        loss = loss_func(pred, yb)
        print(loss.item())
        loss.backward()
        optm.step()
        optm.zero_grad()

print(loss_func(model(xb), yb))

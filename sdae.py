from re import X
import torch
from torch import autograd, Tensor, cuda, optim, nn
from torch._C import set_flush_denormal
from torch.nn import functional as F, ModuleList                 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd, os, argparse
from clearml import Task

from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime as dt
from clearml import Task

device = 'cuda' if cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#task = Task.init(project_name="sdae", task_name="train")


parser = argparse.ArgumentParser()
parser.add_argument('--batch', help='integer value', type=int, default=5)
parser.add_argument('--epoch', help='integer value', type=int, default=5)
parser.add_argument('--lr', help='integer value', type=int, default=5)
parser.add_argument('--run', help='string value', default='211390805')
parser.add_argument('--disable_cuda', help='when you are poor', default=0)

args = parser.parse_args()
print(
    DATA_SET := "gene_exp_MA.csv",
    RANDOM_STATE := 20202,
    DATA_DIR := 'data',
    run_id := dt.strftime(dt.now(),"%y%j%H%M"),
    DATA_DIR := os.path.join(DATA_DIR, run_id)
)

# Loading data
print("Load data...")
raw_df = pd.read_csv(DATA_SET, header=0, sep=",", skipinitialspace=True, index_col=0)

# convert diagnosis attribute to binary format
#raw_df['class_gbm_1'] = raw_df['class_gbm_1'].map({'1': 1, '0': 0})

print("Defining cross validation splits.")
train_df, test_df = train_test_split(raw_df, train_size=0.8, random_state=RANDOM_STATE)    
print("Completed data preparation.")

class AE(nn.Module):
    def __init__(self, df: pd.DataFrame, input_sizes: list = [2000,1000,500]):
        super().__init__()

        self.input_sizes = input_sizes.push(df.shape[1])
        print("Datas has {0} features for {1} samples and {3}.".format(input_sizes[0], df.shape[0],  len(df[1]-1)))
        
        print("Building encoer layers")
        self.fce = nn.ModuleList(map(lambda x: nn.Linear(self.input_sizes)))
        
        print("Building docoder layers")
        self.output_sizes = self.input_sizes.reversed
        self.fcd = nn.ModuleList(map(lambda x: nn.Linear(self.output_sizes)))
        
    def forward_layerwise(self, x_in):

        self.runin = [x_in]
        for i in self.input_size - 1:
            print("Training layer {%d}".format(i))
            x = self.fce[i](self.runin[i])
            self.runin.apppend(x)
            x = self.fcd[i](x)
        return x

    def forward(self, x):
        latent = x
        for enc, dec in zip(self.fce, self.fcd):
            
            latent = enc(x)
            x = dec(latent)

            return x, latent

    def predict(self, x):
        for layer in self.fce:
            x = layer(x)
        x = nn.Linear(x.shape[1], 1)
        

model = AE(train.shape[1]).to(device)
loss_func = torch.nn.MSELoss()
optm=optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epoch):
   for x in train_dl:
        print(type(x[0]))
        pred = model(x)
        print(pred)
        loss = loss_func(pred, x)
        print(loss.item())
        
        optm.zero_grad()
        loss.backward()
        optm.step()
        

print(loss_func(model(*x), *x))

import torch

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameters for gnn
all_epoches = range(200)
n_epochs = 1
# lr = 0.000005
lr = 0.001
wd = 0.1

# negative sample hyperparameters
epsilon = 0.1
proportion = 1

    
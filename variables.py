import torch

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameters for gnn
all_epoches = range(400)
n_epochs = 1
lr = 0.00001
wd = 0.1

# negative sample hyperparameters
epsilon = 0.1
proportion = 1

    
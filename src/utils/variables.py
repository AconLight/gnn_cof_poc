import torch

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameters for gnn
all_epoches = range(600)
n_epochs = 4
# lr = 0.000005
# lr = 0.00001 # ten byl ostatnio
lr = 0.00001
wd = 0.1

# negative sample hyperparameters
epsilon = 0.1
proportion = 0.1

    
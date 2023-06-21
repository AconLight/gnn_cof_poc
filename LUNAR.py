# imports
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import utils
import variables as var
from torch_geometric.nn import MessagePassing
from copy import deepcopy
from sklearn.metrics import precision_score, confusion_matrix
import numpy as np
# Message passing scheme
class GNN1(MessagePassing):
    def __init__(self,k):
        super(GNN1, self).__init__(flow="target_to_source")
        self.k = k
        self.hidden_size = 128#256
        self.network = nn.Sequential(
            nn.Linear(k*2,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,1),
            nn.Sigmoid()
            )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, k = self.k, network=self.network)
        return out

    def message(self,x_i,x_j,edge_attr):
        # message is the edge weight
        return edge_attr

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages
        self.input_aggr = inputs.reshape(-1,k*2)
        # pass through network
        out = self.network(self.input_aggr)
        return out

# GNN
class GNN(torch.nn.Module):
    def __init__(self, k):
        super(GNN, self).__init__()
        self.k = k
        self.L1 = GNN1(self.k)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out
    
def run(dataset,seed,k,samples,train_new_model, data, train_mask, val_mask, test_mask):
    # print('train_new_model: ' + str(train_new_model))
    # loss function
    criterion = nn.MSELoss(reduction = 'none')    

    # path to save model parameters
    model_path = 'saved_models/%s/%d/net_%d.pth' %(dataset,k,seed)
    if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path)) 
    
    # x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, dist2, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon, mode)
    # data = utils.build_graph(x, y, idx, dist, dist2)
        
    data = data.to(var.device)                                                                    
    torch.manual_seed(seed)
    net = GNN(k).to(var.device)
    outs = []
    all_outs = []
    train_outs = []

    test_tps = []
    test_tns = []
    test_fps = []
    test_fns = []
    tps = []
    tns = []
    fps = []
    fns = []


    for epoches_ctr in var.all_epoches:

        optimizer = optim.Adam(net.parameters(), lr = var.lr, weight_decay = var.wd)

        # training
        for epoch in range(var.n_epochs):
            # print('epoch: ' + str(epoch))
            net.train()
            optimizer.zero_grad()
            out = net(data)
            # loss for training data only
            loss = criterion(out[train_mask == 1],data.y[train_mask == 1]).sum()
            loss.backward()
            optimizer.step()


        # testing
        with torch.no_grad():
            net.eval()
            out = net(data)
            loss = criterion(out,data.y)

        # outs.append(roc_auc_score(data.y[test_mask==1].cpu(), out[test_mask==1].cpu()))
        # train_outs.append(roc_auc_score(data.y[train_mask == 1].cpu(), out[train_mask == 1].cpu()))
        # all_outs.append(roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu()))
        out21 = np.round(np.clip(out[test_mask==1].cpu().tolist(), 0, 1)).astype(bool)
        out22 = np.round(np.clip(out[train_mask==1].cpu().tolist(), 0, 1)).astype(bool)
        out23 = np.round(np.clip(out[val_mask==1].cpu().tolist(), 0, 1)).astype(bool)
        outs.append(precision_score(data.y[test_mask==1].cpu().tolist(), out21))
        train_outs.append(precision_score(data.y[train_mask == 1].cpu().tolist(), out22))
        all_outs.append(precision_score(data.y[val_mask==1].cpu().tolist(),out23))

        test_tn, test_fp, test_fn, test_tp = confusion_matrix(data.y[test_mask==1].cpu().tolist(), out21).ravel()
        tn, fp, fn, tp = confusion_matrix(data.y[train_mask == 1].cpu().tolist(), out22).ravel()
        test_tps.append(test_tp)
        test_tns.append(test_tn)
        test_fps.append(test_fp)
        test_fns.append(test_fn)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

        # if len(outs) > 100 and np.average(train_outs[-100:-1]) - 0.15 < np.average(outs[-100:-1]) < np.average(train_outs[-100:-1]) - 0.03:
        #     break
       
    # return output for test points
    return outs, train_outs, all_outs, test_tps, test_tns, test_fps, test_fns, tps, tns, fps, fns

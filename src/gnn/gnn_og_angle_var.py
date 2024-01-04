import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.utils.angle import calc_pair_angles, calc_pair_angles_and_dist, calc_dist, calc_angles_var_and_dist


class GNN1AngleOgVar(MessagePassing):
    def __init__(self,k, input_size):
        super(GNN1AngleOgVar, self).__init__(flow="target_to_source")
        self.is_train = True
        self.input_aggr_train = None
        self.input_aggr_test = None
        self.k = k
        self.input_size = input_size
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(input_size,self.hidden_size),
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

        if self.is_train:
            if self.input_aggr_train is not None:
                out = self.network(self.input_aggr_train)
                return out

            self.input_aggr_train = calc_angles_var_and_dist(inputs, self.k)
            out = self.network(self.input_aggr_train)
            return out
        else:
            if self.input_aggr_test is not None:
                out = self.network(self.input_aggr_test)
                return out

            self.input_aggr_test = calc_angles_var_and_dist(inputs, self.k)
            out = self.network(self.input_aggr_test)
            return out

# GNN
class GNNAngleOgVar(torch.nn.Module):
    def __init__(self, k, input_size):
        super(GNNAngleOgVar, self).__init__()
        self.k = k
        self.L1 = GNN1AngleOgVar(self.k, input_size)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out



import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class GNN1(MessagePassing):
    def __init__(self ,k, input_size, output_size):
        super(GNN1, self).__init__(flow="target_to_source")
        self.k = k
        self.input_size = input_size
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(input_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,output_size),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, k = self.k, network=self.network)
        return out

    def message(self ,x_i ,x_j ,edge_attr):
        # message is the edge weight
        return edge_attr

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages
        self.input_aggr = inputs.reshape(-1 ,self.input_size)
        # pass through network
        out = self.network(self.input_aggr)
        return out


class GNN2(MessagePassing):
    def __init__(self ,k, input_size, output_size):
        super(GNN2, self).__init__(flow="target_to_source")
        self.isTrain = True
        self.k = k
        self.input_size = input_size
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(input_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size ,output_size),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, k = self.k, network=self.network)
        return out

    def message(self ,x_i ,x_j ,edge_attr):
        # message is the edge weight
        return x_j

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages
        self.input_aggr = inputs.reshape(-1 ,self.input_size)
        # pass through network
        out = self.network(self.input_aggr)
        return out

# GNN
class GNNLayer(torch.nn.Module):
    def __init__(self, k, input_size, mid_layer_size = 5):
        super(GNNLayer, self).__init__()
        self.k = k
        self.mid_layer_size = mid_layer_size
        self.L1 = GNN1(self.k, input_size, self.mid_layer_size)
        self.L2 = GNN2(self.k, self.k*self.mid_layer_size, 1)
    def forward(self ,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out1 = self.L1(self.x, self.edge_index, self.edge_attr)
        out2 = self.L2(out1, self.edge_index, self.edge_attr)
        out3 = torch.squeeze(out2 ,1)
        return out3

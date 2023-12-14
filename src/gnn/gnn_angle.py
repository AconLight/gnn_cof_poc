import torch
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.utils import utils


class GNN1AngleFit(MessagePassing):
    def __init__(self,k, input_size):
        super(GNN1AngleFit, self).__init__(flow="target_to_source")
        self.is_train = True
        self.input_aggr_train = None
        self.input_aggr_test = None
        self.k = k
        self.input_size = input_size
        self.hidden_size = 128#256
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
        # concatenate all k messages
        # self.input_aggr = inputs.reshape(-1,self.input_size)
        # my_input_aggr = inputs.reshape(-1,self.input_size)
        # print('aggregate')
        if self.is_train:
            if self.input_aggr_train is not None:
                out = self.network(self.input_aggr_train)
                return out


            message = []
            splited = np.split(inputs, int(inputs.shape[0] / self.k))
            for neighbors in splited:
                pairs = np.split(neighbors, int(self.k / 2))
                message_neighbors = []
                for pair in pairs:
                    angle = utils.angle_between(pair[0], pair[1])
                    message_neighbors.append(angle)
                message.append(message_neighbors)

            self.input_aggr_train = torch.from_numpy(np.array(message)).to(torch.float32)

            out = self.network(self.input_aggr_train)
            return out
        else:
            if self.input_aggr_test is not None:
                out = self.network(self.input_aggr_test)
                return out

            message = []
            splited = np.split(inputs, int(inputs.shape[0] / self.k))
            for neighbors in splited:
                pairs = np.split(neighbors, int(self.k / 2))
                message_neighbors = []
                for pair in pairs:
                    angle = utils.angle_between(pair[0], pair[1])
                    message_neighbors.append(angle)
                message.append(message_neighbors)

            self.input_aggr_test = torch.from_numpy(np.array(message)).to(torch.float32)

            out = self.network(self.input_aggr_test)
            return out

# GNN
class GNNAngleFit(torch.nn.Module):
    def __init__(self, k, input_size):
        super(GNNAngleFit, self).__init__()
        self.k = k
        self.L1 = GNN1AngleFit(self.k, input_size)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out




class GNN1Angle(MessagePassing):
    def __init__(self,k, input_size):
        super(GNN1Angle, self).__init__(flow="target_to_source")
        self.is_train = True
        self.input_aggr_train = None
        self.input_aggr_test = None
        self.k = k
        self.input_size = input_size
        self.hidden_size = 128#256
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
        # concatenate all k messages
        # self.input_aggr = inputs.reshape(-1,self.input_size)
        # my_input_aggr = inputs.reshape(-1,self.input_size)
        # print('aggregate')
        if self.is_train:
            if self.input_aggr_train is not None:
                out = self.network(self.input_aggr_train)
                return out


            message = []
            splited = np.split(inputs, int(inputs.shape[0] / self.k))
            for neighbors in splited:
                pairs = [(a, b) for idx, a in enumerate(neighbors) for b in neighbors[idx + 1:]]
                message_neighbors = []
                for pair in pairs:
                    angle = utils.angle_between(pair[0], pair[1])
                    message_neighbors.append(angle)
                message.append(message_neighbors)

            self.input_aggr_train = torch.from_numpy(np.array(message)).to(torch.float32)

            out = self.network(self.input_aggr_train)
            return out
        else:
            if self.input_aggr_test is not None:
                out = self.network(self.input_aggr_test)
                return out

            message = []
            splited = np.split(inputs, int(inputs.shape[0] / self.k))
            for neighbors in splited:
                pairs = [(a, b) for idx, a in enumerate(neighbors) for b in neighbors[idx + 1:]]
                message_neighbors = []
                for pair in pairs:
                    angle = utils.angle_between(pair[0], pair[1])
                    message_neighbors.append(angle)
                message.append(message_neighbors)

            self.input_aggr_test = torch.from_numpy(np.array(message)).to(torch.float32)

            out = self.network(self.input_aggr_test)
            return out

# GNN
class GNNAngle(torch.nn.Module):
    def __init__(self, k, input_size):
        super(GNNAngle, self).__init__()
        self.k = k
        self.L1 = GNN1Angle(self.k, input_size)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out



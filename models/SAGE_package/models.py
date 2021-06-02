import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, SparseGraphConvolution, SparseLayer, DenseLayer
import torch
import math
from torch.nn.parameter import Parameter

class GNN(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nclass, dropout):
        super(GNN, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dropout = dropout

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                layer_ = SparseLayer(self.dim[i], self.dim[i+1])
                self.layer.append(layer_)
            else:
                layer_ = DenseLayer(self.dim[i], self.dim[i + 1])
                self.layer.append(layer_)

        # self.lin1 = nn.Linear(self.hidden[-1], self.num_classes)

    def forward(self, x):
        for layer_ in range(self.num_layer):
            next_hidden = []
            layer = self.layer[layer_]
            for i in range(self.num_layer - layer_):
                self_feat = x[i]
                neigh_feat = x[i+1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
                if layer_ == self.num_layer - 2:
                    hidden_ = F.relu(hidden_)
                    hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
                next_hidden.append(hidden_)
            x = next_hidden
        output = x[0]
        return F.log_softmax(output, dim=1)

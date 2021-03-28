import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, SparseGraphConvolution, SparseLayer, DenseLayer
import torch
import math
from torch.nn.parameter import Parameter

class GCN_original(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_original, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


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


class GCR(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, dropout):
        super(GCR, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)

        self.num_layer_bi = len(self.hidden_bi)
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
        # bipartite graph layer
        self.layer_2 = nn.ModuleList()
        for i in range(self.num_layer_bi):
            if i == 0:
                layer_ = DenseLayer(self.dim_bi[i], self.dim_bi[i + 1])
                self.layer_2.append(layer_)
            else:
                layer_ = DenseLayer(self.dim_bi[i], self.dim_bi[i + 1])
                self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        self.lin1 = nn.Linear(self.dim[-1] + self.dim_bi[-1], self.num_classes)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                x_bi[i] = F.relu(x_emb)
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_bi[i] = x_emb
        return x_bi

    def forward(self, x, x_bi):
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
                else:
                    hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x = next_hidden
        output_1 = x[0]

        x_bi = self.table_embedding(x_bi)

        for layer_ in range(self.num_layer_bi):
            next_hidden = []
            layer = self.layer_2[layer_]
            for i in range(self.num_layer_bi - layer_):
                self_feat = x_bi[i]
                neigh_feat = x_bi[i+1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num_bi[i])
                if layer_ == self.num_layer_bi - 2:
                    hidden_ = F.relu(hidden_)
                    hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
                else:
                    hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x_bi = next_hidden
        output_2 = x_bi[0]

        output = torch.cat([output_1, output_2], dim=1)
        output = self.lin1(output)

        return F.log_softmax(output, dim=1)

class GCN_3(nn.Module):
    def __init__(self, nfeat, nhid,nclass, dropout):
        super(GCN_3, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid,nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_4(nn.Module):
    def __init__(self, nfeat, nhid,nclass, dropout):
        super(GCN_4, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid,nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_5(nn.Module):
    def __init__(self, nfeat, nhid,nclass, dropout):
        super(GCN_5, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid,nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_6(nn.Module):
    def __init__(self, nfeat, nhid,nclass, dropout):
        super(GCN_6, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid,nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return F.log_softmax(x, dim=1)

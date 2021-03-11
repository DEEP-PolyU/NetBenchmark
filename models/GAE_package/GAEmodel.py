import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, SparseGraphConvolution, SparseLayer, DenseLayer, InnerProductDecoder, NeighLayer, SparseGCN, GCNTrans
import torch
import math
from torch.nn.parameter import Parameter

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCN, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb):
        preds = self.decoder(input_emb)
        return preds

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

class GCNTra(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCNTra, self).__init__()

        self.gc1 = SparseGCN(nfeat, nhid)
        # self.gc1 = GCNTrans(nfeat, nhid)
        self.gc2 = GCNTrans(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb):
        preds = self.decoder(input_emb)
        return preds

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

class GCNTraE(nn.Module):
    def __init__(self, nfeat, nnode, nattri, nhid, nhid2, dropout, act='relu'):
        super(GCNTraE, self).__init__()
        self.num_node = nnode
        self.num_attri = nattri
        self.latent_dim = nfeat
        self.gc1 = GCNTrans(nfeat, nhid)
        self.gc2 = GCNTrans(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout

        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()

        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)
        # self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj):
        x = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        x = self.act(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return self.act(x)

    def pred_logits(self, input_emb):
        preds = self.decoder(input_emb)
        return preds

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

class GCNdense(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCNdense, self).__init__()

        self.gc1 = GCNTrans(nfeat, nhid)
        self.gc2 = GCNTrans(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb):
        preds = self.decoder(input_emb)
        return preds

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

class GCNSample(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCNSample, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        # self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        pos_src_emb = input_emb[pos_src]
        pos_dst_emb = input_emb[pos_dst]
        neg_src_emb = input_emb[neg_src]
        neg_dst_emb = input_emb[neg_dst]
        pos_logit = torch.sum(torch.mul(pos_src_emb, pos_dst_emb), dim=1)
        neg_logit = torch.sum(torch.mul(neg_src_emb, neg_dst_emb), dim=1)
        return pos_logit, neg_logit

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

class GCNANE(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nnode, nattri, dropout):
        super(GCNANE, self).__init__()
        self.latent_dim = nfeat
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)

        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj):
        if self.training:
            adj = self.__dropout(adj)
        x = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb):
        logits = self.decoder(input_emb)
        return logits

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        values = values[random_index]/self.dropout
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss


class GCNANESample(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, nnode, nattri, dropout):
        super(GCNANESample, self).__init__()
        self.latent_dim = nfeat
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid2)
        self.decoder = InnerProductDecoder(nhid2, dropout)
        self.dropout = dropout
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)

        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj):
        if self.training:
            adj = self.__dropout(adj)
        x = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        pos_src_emb = input_emb[pos_src]
        pos_dst_emb = input_emb[pos_dst]
        neg_src_emb = input_emb[neg_src]
        neg_dst_emb = input_emb[neg_dst]
        pos_logit = torch.sum(torch.mul(pos_src_emb, pos_dst_emb), dim=1)
        neg_logit = torch.sum(torch.mul(neg_src_emb, neg_dst_emb), dim=1)
        return pos_logit, neg_logit

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        values = values[random_index]/self.dropout
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss

class GCRANEHid(nn.Module):
    def __init__(self, nfeat, ndim, agg_type, nnode, nattri, dropout, drop, hid1=512, hid2=128):
        super(GCRANEHid, self).__init__()
        self.n_dim = ndim
        self.latent_dim = nfeat
        self.gc1 = NeighLayer(ndim, ndim, agg_type)
        self.gc2 = NeighLayer(ndim, ndim, agg_type)
        self.decoder = InnerProductDecoder(ndim, dropout)
        self.dropout = dropout
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)

        # Layer-wise feature transformation
        self.trans1 = nn.Linear(self.latent_dim, self.n_dim, bias=False)
        self.trans2 = nn.Linear(self.latent_dim, self.n_dim, bias=False)
        self.trans3 = nn.Linear(self.latent_dim, self.n_dim, bias=False)
        # prediction mlp layer
        self.mlp1 = nn.Linear(9 * self.n_dim, self.hid1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj, adj2):
        if self.training:
            if self.drop:
                adj = self.__dropout(adj)
                adj2 = self.__dropout(adj2)
        x0 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        x1 = self.trans1(x0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x2 = self.gc1(x1, adj)
        x3 = self.gc2(x1, adj2)
        return [x1, x2, x3]

    def get_emb(self):
        node_emb = self.embedding_node.weight
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, x_layer, src, dst):
        src_x = [xx[src] for xx in x_layer]
        dst_x = [xx[dst] for xx in x_layer]

        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = F.relu(emb)
        emb = self.mlp2(emb)
        emb = F.relu(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        emb_pos = self.cross_layer(input_emb, pos_src, pos_dst)
        emb_neg = self.cross_layer(input_emb, neg_src, neg_dst)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss

class GCRANEGNN(nn.Module):
    def __init__(self, nfeat, agg_type, nnode, nattri, dropout, drop, hid1=512, hid2=128):
        super(GCRANEGNN, self).__init__()
        self.latent_dim = nfeat
        self.hid1 = hid1
        self.hid2 = hid2
        self.gc1 = GCNTrans(self.latent_dim, self.hid1)
        self.gc2 = GCNTrans(self.hid1, self.hid2)
        self.decoder = InnerProductDecoder(self.hid2, dropout)
        self.dropout = dropout
        self.drop = drop
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)

        # prediction layer
        # self.mlp1 = nn.Linear(9 * self.latent_dim, self.hid1, bias=False)
        # self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        # self.mlp3 = nn.Linear(self.hid2, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj):
        x = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

    def get_emb(self):
        node_emb = self.embedding_node.weight
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, x_layer, src, dst):
        src_x = [xx[src] for xx in x_layer]
        dst_x = [xx[dst] for xx in x_layer]

        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = F.relu(emb)
        emb = self.mlp2(emb)
        emb = F.relu(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        emb_pos_src = input_emb[pos_src]
        emb_pos_dst = input_emb[pos_dst]
        emb_neg_src = input_emb[neg_src]
        emb_neg_dst = input_emb[neg_dst]
        logits_pos = torch.sum(torch.mul(emb_pos_src, emb_pos_dst), dim=1)
        logits_neg = torch.sum(torch.mul(emb_neg_src, emb_neg_dst), dim=1)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss

class GCRANE(nn.Module):
    def __init__(self, nfeat, agg_type, nnode, nattri, dropout, drop, hid1=512, hid2=128):
        super(GCRANE, self).__init__()
        self.latent_dim = nfeat
        self.gc1 = NeighLayer(nfeat, nfeat, agg_type)
        self.gc2 = NeighLayer(nfeat, nfeat, agg_type)
        self.decoder = InnerProductDecoder(nfeat, dropout)
        self.dropout = dropout
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)

        # prediction layer
        self.mlp1 = nn.Linear(9 * self.latent_dim, self.hid1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj, adj2):
        if self.training:
            if self.drop:
                adj = self.__dropout(adj)
                adj2 = self.__dropout(adj2)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x2 = self.gc1(x1, adj)
        x3 = self.gc2(x1, adj2)
        return [x1, x2, x3]

    def get_emb(self):
        node_emb = self.embedding_node.weight
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, x_layer, src, dst):
        src_x = [xx[src] for xx in x_layer]
        dst_x = [xx[dst] for xx in x_layer]

        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = F.relu(emb)
        emb = self.mlp2(emb)
        emb = F.relu(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        emb_pos = self.cross_layer(input_emb, pos_src, pos_dst)
        emb_neg = self.cross_layer(input_emb, neg_src, neg_dst)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss

class GCRANE3(nn.Module):
    def __init__(self, nfeat, agg_type, nnode, nattri, dropout, drop, hid1=512, hid2=128):
        super(GCRANE3, self).__init__()
        self.latent_dim = nfeat
        self.gc1 = NeighLayer(nfeat, nfeat, agg_type)
        self.gc2 = NeighLayer(nfeat, nfeat, agg_type)
        self.gc3 = NeighLayer(nfeat, nfeat, agg_type)
        self.decoder = InnerProductDecoder(nfeat, dropout)
        self.dropout = dropout
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)

        # prediction layer
        self.mlp1 = nn.Linear(16 * self.latent_dim, self.hid1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.embedding_node.weight)
        # torch.nn.init.xavier_uniform_(self.embedding_attri.weight)
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj, adj2, adj3):
        if self.training:
            if self.drop:
                adj = self.__dropout(adj)
                adj2 = self.__dropout(adj2)
                adj3 = self.__dropout(adj3)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        # x = F.dropout(x, self.dropout, training=self.training)
        x2 = self.gc1(x1, adj)
        x3 = self.gc2(x1, adj2)
        x4 = self.gc3(x1, adj3)
        return [x1, x2, x3, x4]

    def get_emb(self):
        node_emb = self.embedding_node.weight
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, x_layer, src, dst):
        src_x = [xx[src] for xx in x_layer]
        dst_x = [xx[dst] for xx in x_layer]

        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = F.relu(emb)
        emb = self.mlp2(emb)
        emb = F.relu(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, input_emb, pos_src, pos_dst, neg_src, neg_dst):
        emb_pos = self.cross_layer(input_emb, pos_src, pos_dst)
        emb_neg = self.cross_layer(input_emb, neg_src, neg_dst)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss


class GCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2, self).__init__()

        self.gc1 = SparseGraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = SparseGraphConvolution(nfeat, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.lin1 = nn.Linear(nclass * 2, nclass)

    def forward(self, x, adj, adj2, trade_off=.8):
        x2 = x
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        x2 = F.relu(self.gc3(x2, adj2))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gc4(x2, adj2)

        # output = trade_off * x + (1 - trade_off) * x2
        output = torch.cat([x, x2], dim=1)
        weight = F.sigmoid(self.lin1(output))
        output = weight * x + (1 - weight) * x2

        return F.log_softmax(output, dim=1)


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nclass, neg_num, dropout):
        super(GNN, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.num_classes = nclass
        # self.dim[-1] = self.num_classes
        self.dropout = dropout
        self.neg_num = neg_num
        self.pos_weight = torch.FloatTensor([float(self.neg_num)])
        self.norm = float((1.0 + neg_num) / (neg_num * 2))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                layer_ = SparseLayer(self.dim[i], self.dim[i+1])
                self.layer.append(layer_)
            else:
                layer_ = DenseLayer(self.dim[i], self.dim[i + 1])
                self.layer.append(layer_)

        # self.lin1 = nn.Linear(self.hidden[-1], self.num_classes)

    def forward(self, x_src, x_dst, x_neg):
        for layer_ in range(self.num_layer):
            next_hidden = []
            next_hidden_dst = []
            next_hidden_neg = []
            layer = self.layer[layer_]
            for i in range(self.num_layer - layer_):
                self_feat = x_src[i]
                neigh_feat = x_src[i+1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])

                self_feat_dst = x_dst[i]
                neigh_feat_dst = x_dst[i + 1]
                hidden_dst = layer(self_feat_dst, neigh_feat_dst, self.neigh_num[i])

                self_feat_neg = x_neg[i]
                neigh_feat_neg = x_neg[i + 1]
                hidden_neg = layer(self_feat_neg, neigh_feat_neg, self.neigh_num[i])

                if layer_ == self.num_layer - 2:
                    hidden_ = F.relu(hidden_)
                    hidden_ = F.dropout(hidden_, self.dropout, training=self.training)

                    hidden_dst = F.relu(hidden_dst)
                    hidden_dst = F.dropout(hidden_dst, self.dropout, training=self.training)

                    hidden_neg = F.relu(hidden_neg)
                    hidden_neg = F.dropout(hidden_neg, self.dropout, training=self.training)
                else:
                    hidden_ = F.relu(hidden_)

                    hidden_dst = F.relu(hidden_dst)

                    hidden_neg = F.relu(hidden_neg)
                next_hidden.append(hidden_)
                next_hidden_dst.append(hidden_dst)
                next_hidden_neg.append(hidden_neg)
            x_src = next_hidden
            x_dst = next_hidden_dst
            x_neg = next_hidden_neg
        return x_src[0], x_dst[0], x_neg[0]

    def generate_emb(self, x_src):
        for layer_ in range(self.num_layer):
            next_hidden = []
            layer = self.layer[layer_]
            for i in range(self.num_layer - layer_):
                self_feat = x_src[i]
                neigh_feat = x_src[i+1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])

                if layer_ == self.num_layer - 2:
                    hidden_ = F.relu(hidden_)
                else:
                    hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x_src = next_hidden
        return x_src[0]

    def xent_loss(self, x_src, x_dst, x_neg, num_neg):
        pos_logit = self.affinity(x_src, x_dst)
        neg_logit = self.affinity(x_src.view(-1, 1, self.dim[-1]), x_neg.view(-1, num_neg, self.dim[-1]))

        # pos_score = torch.log(torch.sigmoid(pos_logit)).view(-1, 1)
        # neg_score = torch.log(torch.sigmoid(- neg_logit)).view(-1, num_neg)
        # neg_score = torch.mean(torch.log(torch.sigmoid(- neg_logit)), 1)
        # neg_score = num_neg * torch.mean(torch.log(torch.sigmoid(- neg_logit)), 1)
        # pos_score = pos_logit.view(-1, 1)
        # neg_score = neg_logit.view(-1, self.neg_num)
        # logits = torch.cat([pos_score, neg_score], dim=1)
        # labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=1)
        # loss = self.loss_fn(logits, labels)
        # loss *= self.norm

        pos_score = pos_logit.view(-1)
        neg_score = neg_logit.view(-1)

        loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))



        neg_logit = neg_logit.view(-1)
        # preds = torch.cat([pos_logit, neg_logit], dim=-1)
        # label = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=-1)

        # all_score = torch.cat([pos_score, neg_score], dim=1).view(-1)
        # loss_ = torch.mean(- all_score)

        logits = torch.sigmoid(torch.cat([pos_logit, neg_logit], dim=-1))
        label = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=-1)
        # loss = torch.cat([true_xent, negative_xent], dim=1)
        # loss = torch.mean(loss)
        return loss, logits, label

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]
        result = torch.sum(torch.mul(inputs1, inputs2), dim=-1)
        return result

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            # print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            # print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

        loss = torch.mean(torch.cat(nodes_score, 0))

        return loss




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
        self.dim_bi[-1] = self.num_classes
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
        self.reset_parameters()

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

    def forward(self, x, x_bi, com_weight=0.8):
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
                # else:
                #     hidden_ = F.relu(hidden_)
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
                # else:
                #     hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x_bi = next_hidden
        output_2 = x_bi[0]

        # output = torch.cat([output_1, output_2], dim=1)
        output = com_weight * output_1 + (1 - com_weight) * output_2
        # output = output_1 + output_2
        # output = self.lin1(output)

        return F.log_softmax(output, dim=1)

class GCR2(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, atten, dropout):
        super(GCR2, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)
        self.atten = atten
        self.emb_dim = tdim

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout
        self.bi_num = 9

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer.append(layer_)
            else:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer.append(layer_)
        # bipartite graph layer
        self.layer_2 = nn.ModuleList()
        for i in range(self.num_layer_bi):
            if i == 0:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer_2.append(layer_)
            else:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, self.emb_dim))
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, self.emb_dim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                # x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            # x_list[i] = F.relu(x_emb)
            x_list[i] = x_emb
        return x_list

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)
        x1_layer.append(x[0])

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
                # else:
                #     hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x = next_hidden
            x1_layer.append(x[0])

        x_bi = self.table_embedding(x_bi)

        x2_layer = []

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
                # else:
                #     hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x_bi = next_hidden
            x2_layer.append(x_bi[0])

        cross_layer = self.bi_cross_layer(x1_layer, x2_layer)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)
        # output = torch.cat([output_1, output_2], dim=1)
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        # output = output_1 + output_2
        # output = self.lin1(output)

        return F.log_softmax(output, dim=1)


class GCRBi(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, dropout):
        super(GCRBi, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)
        self.emb_dim = tdim
        self.cross_layer = cross_layer

        if self.cross_layer == 'cross':
            self.bi_num = 6
        else:
            self.bi_num = 7

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            if i == 0:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                # layer_ = SparseLayer(self.emb_dim, self.emb_dim)
                self.layer.append(layer_)
            else:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer.append(layer_)
        # bipartite graph layer
        self.layer_2 = nn.ModuleList()
        for i in range(self.num_layer_bi):
            if i == 0:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer_2.append(layer_)
            else:
                layer_ = DenseLayer(self.emb_dim, self.emb_dim)
                self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        self.lin1 = nn.Linear(self.emb_dim + self.bi_num, 1)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                # x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            # x_list[i] = F.relu(x_emb)
            x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):

        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)

        for layer_ in range(self.num_layer):
            next_hidden = []
            layer = self.layer[layer_]
            for i in range(self.num_layer - layer_):
                self_feat = x[i]
                neigh_feat = x[i + 1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
                if layer_ == self.num_layer - 2:
                    hidden_ = F.relu(hidden_)
                    hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
                # else:
                #     hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x = next_hidden
            x1_layer.append(x[0])
        output_1 = x[0]

        x_bi = self.table_embedding(x_bi)

        x2_layer = []
        for layer_ in range(self.num_layer_bi):
            next_hidden = []
            layer = self.layer_2[layer_]
            for i in range(self.num_layer_bi - layer_):
                self_feat = x_bi[i]
                neigh_feat = x_bi[i + 1]
                hidden_ = layer(self_feat, neigh_feat, self.neigh_num_bi[i])
                if layer_ == self.num_layer_bi - 2:
                    hidden_ = F.relu(hidden_)
                    hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
                # else:
                #     hidden_ = F.relu(hidden_)
                next_hidden.append(hidden_)
            x_bi = next_hidden
            x2_layer.append(x_bi[0])
        # output_2 = x_bi[0]

        cross_layer = self.bi_cross_layer(x1_layer, x2_layer)
        if self.cross_layer == 'self':
            cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        cross_layer = torch.cat(cross_layer, dim=1)

        atten = self.lin1(cross_layer)
        atten = F.softmax(atten, dim=1)     # [B, num, 1]
        output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)

        output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(output)

        return F.log_softmax(output, dim=1)


class GCRBi2(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi2, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9
        else:
            self.bi_num = 9

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        # self.layer = nn.ModuleList()
        # for i in range(self.num_layer):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         # layer_ = SparseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        # # bipartite graph layer
        # self.layer_2 = nn.ModuleList()
        # for i in range(self.num_layer_bi):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        # self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        # stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        # self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                # x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                # x_emb = torch.mm(x_emb, self.weight_trans2)
                # x_emb = x_emb.view(-1, self.emb_dim)
                # x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            # x_list[i] = F.relu(x_emb)
            x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)

        # for layer_ in range(self.num_layer):
        #     next_hidden = []
        #     layer = self.layer[layer_]
        #     for i in range(self.num_layer - layer_):
        #         self_feat = x[i]
        #         neigh_feat = x[i + 1]
        #         hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
        #         if layer_ == self.num_layer - 2:
        #             hidden_ = F.relu(hidden_)
        #             hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
        #         next_hidden.append(hidden_)
        #     x = next_hidden
        #     x1_layer.append(x[0])
        # output_1 = x[0]
        x1_layer.append(x[0])
        # x1_layer.append(torch.sum(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.sum(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        temp, _ = torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1)
        x1_layer.append(temp)
        temp, _ = torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1)
        x1_layer.append(temp)

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        temp, _ = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        x_bi[0] = temp
        temp, _ = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        x_bi[1] = temp

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)


class GCRBi2all(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi2all, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9 + 3
        else:
            self.bi_num = 9 + 3

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        # self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        # stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        # self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                # x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                # x_emb = torch.mm(x_emb, self.weight_trans2)
                # x_emb = x_emb.view(-1, self.emb_dim)
                # x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            # x_list[i] = F.relu(x_emb)
            x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)
        x1_layer.append(x[0])
        # x1_layer.append(torch.sum(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.sum(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        x1_layer.append(torch.mean(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        x1_layer.append(torch.mean(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        # temp, _ = torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1)
        # x1_layer.append(temp)
        # temp, _ = torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1)
        # x1_layer.append(temp)

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # temp, _ = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[0] = temp
        # temp, _ = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[1] = temp

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        cross_layer += self.bi_self_layer(x1_layer)
        # cross_layer += self.bi_self_layer(x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)


class GCRBi2atten(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi2atten, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, tdim)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9
        else:
            self.bi_num = 9

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        # self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        self.x_hop1 = nn.Linear(self.emb_dim, 1)
        self.x_hop2 = nn.Linear(self.emb_dim, 1)
        self.a_hop1 = nn.Linear(self.emb_dim, 1)
        self.a_hop2 = nn.Linear(self.emb_dim, 1)
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        # stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        # self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                # x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                # x_emb = torch.mm(x_emb, self.weight_trans2)
                # x_emb = x_emb.view(-1, self.emb_dim)
                # x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            # x_list[i] = F.relu(x_emb)
            x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def atten_layer(self, input, layer, num_neigh):
        input = input.view(-1, num_neigh, self.emb_dim)
        atten = layer(input)    # [B, num_neigh, 1]
        atten = F.softmax(atten, dim=1)
        output = torch.sum(atten * input, dim=1)
        return output

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)
        x1_layer.append(x[0])

        # x1_layer.append(torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))

        x1_layer.append(self.atten_layer(x[1], self.x_hop1, self.neigh_num[0]))
        x1_layer.append(self.atten_layer(x[2], self.x_hop2, self.neigh_num[0] * self.neigh_num[1]))
        # x1_layer.append(torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)

        # x_bi[0] = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        x_bi[0] = self.atten_layer(x_bi[0], self.a_hop1, self.neigh_num_bi[0])
        x_bi[1] = self.atten_layer(x_bi[1], self.a_hop2, self.neigh_num_bi[0] * self.neigh_num_bi[1])

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)

class GCRBi3(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi3, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, 64)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9
        else:
            self.bi_num = 9

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        # self.layer = nn.ModuleList()
        # for i in range(self.num_layer):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         # layer_ = SparseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        # # bipartite graph layer
        # self.layer_2 = nn.ModuleList()
        # for i in range(self.num_layer_bi):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_emb = torch.mm(x_emb, self.weight_trans2)
                x_emb = x_emb.view(-1, self.emb_dim)
                x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            x_list[i] = F.relu(x_emb)
            # x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)

        # for layer_ in range(self.num_layer):
        #     next_hidden = []
        #     layer = self.layer[layer_]
        #     for i in range(self.num_layer - layer_):
        #         self_feat = x[i]
        #         neigh_feat = x[i + 1]
        #         hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
        #         if layer_ == self.num_layer - 2:
        #             hidden_ = F.relu(hidden_)
        #             hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
        #         next_hidden.append(hidden_)
        #     x = next_hidden
        #     x1_layer.append(x[0])
        # output_1 = x[0]
        x1_layer.append(x[0])
        # x1_layer.append(torch.sum(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.sum(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        temp, _ = torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1)
        x1_layer.append(temp)
        temp, _ = torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1)
        x1_layer.append(temp)

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        temp, _ = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        x_bi[0] = temp
        temp, _ = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        x_bi[1] = temp

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)


class GCRBi3all(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi3all, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, 64)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9 + 3
        else:
            self.bi_num = 9 + 3

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        # self.layer = nn.ModuleList()
        # for i in range(self.num_layer):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         # layer_ = SparseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        # # bipartite graph layer
        # self.layer_2 = nn.ModuleList()
        # for i in range(self.num_layer_bi):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_emb = torch.mm(x_emb, self.weight_trans2)
                x_emb = x_emb.view(-1, self.emb_dim)
                x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            x_list[i] = F.relu(x_emb)
            # x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)

        # for layer_ in range(self.num_layer):
        #     next_hidden = []
        #     layer = self.layer[layer_]
        #     for i in range(self.num_layer - layer_):
        #         self_feat = x[i]
        #         neigh_feat = x[i + 1]
        #         hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
        #         if layer_ == self.num_layer - 2:
        #             hidden_ = F.relu(hidden_)
        #             hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
        #         next_hidden.append(hidden_)
        #     x = next_hidden
        #     x1_layer.append(x[0])
        # output_1 = x[0]
        x1_layer.append(x[0])
        # x1_layer.append(torch.sum(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.sum(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        x1_layer.append(torch.mean(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        x1_layer.append(torch.mean(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        # temp, _ = torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1)
        # x1_layer.append(temp)
        # temp, _ = torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1)
        # x1_layer.append(temp)

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # temp, _ = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[0] = temp
        # temp, _ = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[1] = temp

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        cross_layer += self.bi_self_layer(x1_layer)
        # cross_layer += self.bi_self_layer(x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)

class GCRBi3atten(nn.Module):
    def __init__(self, nfeat, nhid, ndim, nhid_bi, ndim_bi, tdim, tnum, nclass, cross_layer, atten, dropout):
        super(GCRBi3atten, self).__init__()
        self.hidden = [] if nhid == '' else [int(x) for x in nhid.split(',')]
        self.neigh_num = [] if ndim == '' else [int(x) for x in ndim.split(',')]
        self.hidden_bi = [] if nhid_bi == '' else [int(x) for x in nhid_bi.split(',')]
        self.neigh_num_bi = [] if ndim_bi == '' else [int(x) for x in ndim_bi.split(',')]
        self.num_layer = len(self.hidden)
        self.dim = [nfeat] + self.hidden
        self.dim_bi = [tdim] + self.hidden_bi
        self.table = nn.Embedding(tnum, 64)
        self.emb_dim = tdim
        self.cross_layer = cross_layer
        self.atten = atten

        if self.cross_layer == 'cross':
            self.bi_num = 9
        else:
            self.bi_num = 9

        self.num_layer_bi = len(self.hidden_bi)
        self.num_classes = nclass
        self.dim[-1] = self.num_classes
        self.dim_bi[-1] = self.num_classes
        self.dropout = dropout

        # self.layer = nn.ModuleList()
        # for i in range(self.num_layer):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         # layer_ = SparseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer.append(layer_)
        # # bipartite graph layer
        # self.layer_2 = nn.ModuleList()
        # for i in range(self.num_layer_bi):
        #     if i == 0:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)
        #     else:
        #         layer_ = DenseLayer(self.emb_dim, self.emb_dim)
        #         self.layer_2.append(layer_)

        self.weight_trans = Parameter(torch.FloatTensor(nfeat, tdim))
        self.weight_trans2 = Parameter(torch.FloatTensor(64, tdim))
        self.x_hop1 = nn.Linear(self.emb_dim, 1)
        self.x_hop2 = nn.Linear(self.emb_dim, 1)
        self.a_hop1 = nn.Linear(self.emb_dim, 1)
        self.a_hop2 = nn.Linear(self.emb_dim, 1)
        if self.atten == 'atten':
            self.lin1 = nn.Linear(self.emb_dim, 1)
        else:
            self.lin1 = nn.Linear(self.emb_dim * self.bi_num, tdim)
        self.lin2 = nn.Linear(self.emb_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_trans.size(1))
        self.weight_trans.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_trans2.size(1))
        self.weight_trans2.data.uniform_(-stdv, stdv)

    def table_embedding(self, x_bi):
        for i, x in enumerate(x_bi):
            if (i + 1) % 2 == 1:
                x_emb = torch.spmm(x, self.weight_trans)
                x_bi[i] = F.relu(x_emb)
                x_bi[i] = x_emb
            else:
                x_emb = self.table(x)
                x_emb = x_emb.view(-1, x_emb.size(-1))
                x_emb = torch.mm(x_emb, self.weight_trans2)
                x_emb = x_emb.view(-1, self.emb_dim)
                x_emb = F.relu(x_emb)
                x_bi[i] = x_emb
        return x_bi

    def emb_transform(self, x_list):
        for i, x in enumerate(x_list):
            x_emb = torch.spmm(x, self.weight_trans)
            x_list[i] = F.relu(x_emb)
            # x_list[i] = x_emb
        return x_list

    def bi_self_layer(self, x_layer):
        bi_layer = []
        for i in range(len(x_layer)):
            xi = x_layer[i]
            for j in range(i+1, len(x_layer)):
                xj = x_layer[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def atten_layer(self, input, layer, num_neigh):
        input = input.view(-1, num_neigh, self.emb_dim)
        atten = layer(input)    # [B, num_neigh, 1]
        atten = F.softmax(atten, dim=1)
        output = torch.sum(atten * input, dim=1)
        return output


    def forward(self, x, x_bi, com_weight=0.8):
        x1_layer = []
        x = self.emb_transform(x)

        # for layer_ in range(self.num_layer):
        #     next_hidden = []
        #     layer = self.layer[layer_]
        #     for i in range(self.num_layer - layer_):
        #         self_feat = x[i]
        #         neigh_feat = x[i + 1]
        #         hidden_ = layer(self_feat, neigh_feat, self.neigh_num[i])
        #         if layer_ == self.num_layer - 2:
        #             hidden_ = F.relu(hidden_)
        #             hidden_ = F.dropout(hidden_, self.dropout, training=self.training)
        #         next_hidden.append(hidden_)
        #     x = next_hidden
        #     x1_layer.append(x[0])
        # output_1 = x[0]
        x1_layer.append(x[0])
        # x1_layer.append(torch.sum(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.sum(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.mean(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))

        # x1_layer.append(torch.max(x[1].view(-1, self.neigh_num[0], self.emb_dim), dim=1))
        # x1_layer.append(torch.max(x[2].view(-1, self.neigh_num[1] * self.neigh_num[0], self.emb_dim), dim=1))
        x1_layer.append(self.atten_layer(x[1], self.x_hop1, self.neigh_num[0]))
        x1_layer.append(self.atten_layer(x[2], self.x_hop2, self.neigh_num[0] * self.neigh_num[1]))

        x_bi = self.table_embedding(x_bi)
        x_bi = x_bi[1:]

        # x_bi[0] = torch.sum(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.sum(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)
        # x_bi[0] = torch.mean(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.mean(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)

        # x_bi[0] = torch.max(x_bi[0].view(-1, self.neigh_num_bi[0], self.emb_dim), dim=1)
        # x_bi[1] = torch.max(x_bi[1].view(-1, self.neigh_num_bi[0] * self.neigh_num_bi[1], self.emb_dim), dim=1)

        x_bi[0] = self.atten_layer(x_bi[0], self.a_hop1, self.neigh_num_bi[0])
        x_bi[1] = self.atten_layer(x_bi[1], self.a_hop2, self.neigh_num_bi[0] * self.neigh_num_bi[1])

        cross_layer = self.bi_cross_layer(x1_layer, x_bi)
        # if self.cross_layer == 'self':
        #     cross_layer = cross_layer + self.bi_self_layer(x1_layer)
        cross_layer = cross_layer + x1_layer

        if self.atten == 'atten':
            cross_layer = torch.stack(cross_layer, dim=1)
            atten_ = self.lin1(cross_layer)
            atten_ = F.softmax(atten_, dim=1)
            hidden = torch.sum(torch.mul(atten_, cross_layer), dim=1)
        else:
            cross_layer = torch.cat(cross_layer, dim=1)
            hidden = self.lin1(cross_layer)
            hidden = F.relu(hidden)
        # atten = F.softmax(atten, dim=1)     # [B, num, 1]
        # output_2 = torch.sum(torch.mul(atten, cross_layer), dim=1)
        #
        # output = com_weight * output_1 + (1 - com_weight) * output_2
        output = self.lin2(hidden)

        return F.log_softmax(output, dim=1)
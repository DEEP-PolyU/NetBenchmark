import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .model import *

class SDNE_layer(nn.Module):
    def __init__(self, num_node, hidden_size1, hidden_size2, droput, alpha, beta, nu1, nu2):
        super(SDNE_layer, self).__init__()
        self.num_node = num_node
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.droput = droput
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        self.encode0 = nn.Linear(self.num_node, self.hidden_size1)
        self.encode1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.decode0 = nn.Linear(self.hidden_size2, self.hidden_size1)
        self.decode1 = nn.Linear(self.hidden_size1, self.num_node)

    def forward(self, adj_mat, l_mat):
        t0 = F.leaky_relu(self.encode0(adj_mat))
        t0 = F.leaky_relu(self.encode1(t0))
        self.embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))

        L_1st = 2 * torch.trace(torch.mm(torch.mm(torch.t(self.embedding), l_mat), self.embedding))
        L_2nd = torch.sum(((adj_mat - t0) * adj_mat * self.beta) * ((adj_mat - t0) * adj_mat * self.beta))

        L_reg = 0
        for param in self.parameters():
            L_reg += self.nu1 * torch.sum(torch.abs(param)) + self.nu2 * torch.sum(param * param)
        return self.alpha * L_1st, L_2nd, self.alpha * L_1st + L_2nd, L_reg

    def get_emb(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0




class SDNE(Models):
    def check_train_parameters(self):
        space_dtree = {
            'beta': hp.uniform('beta', 0, 1),
            'evaluation': str(self.evaluation)
        }

        return space_dtree

    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return False

    @classmethod
    def is_end2end(cls):
        return False

    def check_train_parameters(self):
        space_dtree = {
            'batch_size': hp.uniformint('batch_size', 1, 100),
            'nb_epochs': hp.uniformint('nb_epochs', 100, 5000),
            # 'lr': hp.loguniform('lr', np.log(0.05), np.log(0.2)), # walk_length,window_size
            'lr': hp.choice('lr', [0, 1, 2, 3, 4, 5, 6]),
            'dropout': hp.uniform('dropout', 0, 0.75),
            'evaluation': str(self.evaluation)
        }

        return space_dtree

    def train_model(self, **kwargs):
        np.random.seed(42)
        torch.manual_seed(42)
        lrrate = [0.1, 0.01, 0.001, 0.0001, 0.005, 0.05, 0.00005]

        nb_epochs = int(kwargs["nb_epochs"])

        lr = kwargs["lr"]
        lr = lrrate[lr]
        l2_coef = 0.0
        drop_prob = kwargs["dropout"]
        hid_units = 128


        self.graph = self.mat_content['Network']
        G = nx.from_scipy_sparse_matrix(self.graph)
        num_node = G.number_of_nodes()
        #TODO: Parameter range of alpha ,beta, nu1,nu2
        model = SDNE_layer(
            num_node, hidden_size1=256, hidden_size2=128, droput=drop_prob, alpha=0.1, beta=5, nu1=1e-4, nu2=1e-3
        )

        A = torch.from_numpy(nx.adjacency_matrix(G).todense().astype(np.float32))
        L = torch.from_numpy(nx.laplacian_matrix(G).todense().astype(np.float32))

        A, L = A.to(self.device), L.to(self.device)
        model = model.to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_iter = tqdm(range(nb_epochs))
        for epoch in epoch_iter:
            opt.zero_grad()
            L_1st, L_2nd, L_all, L_reg = model.forward(A, L)
            Loss = L_all + L_reg
            Loss.backward()
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, L_1st: {L_1st:.4f}, L_2nd: {L_2nd:.4f}, L_reg: {L_reg:.4f}"
            )
            opt.step()
        embedding = model.get_emb(A)
        return embedding.detach().cpu().numpy()
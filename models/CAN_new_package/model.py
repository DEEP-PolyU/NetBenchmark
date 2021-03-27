from .layers  import GraphConvolution, GraphConvolutionSparse, InnerDecoder, Dense
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class CAN_original(nn.Module):

    def __init__(self,hidden1, hidden2, num_features, num_nodes, features_nonzero,dropout):
        super(CAN_original, self).__init__()
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.dropout = dropout
        '''init里定义的这些layer的参数都是传到对应layer的init处'''
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,output_dim=hidden1,dropout=self.dropout,features_nonzero=self.features_nonzero)
                                              
        self.hidden2 = Dense(input_dim=self.n_samples,output_dim=hidden1,sparse_inputs=True)

        self.z_u_mean = GraphConvolution(input_dim=hidden1,output_dim=hidden2,dropout=self.dropout)

        self.z_u_log_std = GraphConvolution(input_dim=hidden1,output_dim=hidden2,dropout=self.dropout)

        self.z_a_mean = Dense(input_dim=hidden1,output_dim=hidden2,dropout=self.dropout)

        self.z_a_log_std = Dense(input_dim=hidden1,output_dim=hidden2,dropout=self.dropout)

        self.reconstructions = InnerDecoder(input_dim=hidden2)

    '''模型本质就是两个VAE,一个VAE对x和adj的乘积做encoding生成均值z_u_mean,方差(这里是方差的log)z_u_log_std，做decoding生成preds_sub_u
    另一个对x的转置做生成z_a_mean,z_a_log_std和preds_sub_a'''
    def encode(self, x,adj):
        '''在forward中调用的layer的参数传到对应layer的forward处'''
        z_u = F.relu(self.hidden1(x,adj))
        z_a = torch.tanh(self.hidden2(x.t()))
        return self.z_u_mean(z_u,adj), self.z_u_log_std(z_u,adj),self.z_a_mean(z_a), self.z_a_log_std(z_a)
    
    def reparameterize(self, z_u_mean,z_u_log_std,z_a_mean,z_a_log_std): 
        z_u = z_u_mean + torch.randn_like(z_u_log_std) * torch.exp(z_u_log_std)
        z_a = z_a_mean + torch.randn_like(z_a_log_std) * torch.exp(z_a_log_std)
        return z_u,z_a

    def decode(self, z_u,z_a):
        return self.reconstructions (z_u, z_a)
    
    def forward(self, features,adj):
        z_u_mean,z_u_log_std,z_a_mean,z_a_log_std = self.encode(features,adj)
        z_u,z_a = self.reparameterize(z_u_mean,z_u_log_std,z_a_mean,z_a_log_std)
        preds_sub_u, preds_sub_a = self.decode(z_u,z_a)
        return preds_sub_u, preds_sub_a,z_u_mean,z_u_log_std,z_a_mean,z_a_log_std


class CAN_3(nn.Module):

    def __init__(self, hidden1, hidden2, num_features, num_nodes, features_nonzero, dropout):
        super(CAN_3, self).__init__()
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.dropout = dropout
        '''init里定义的这些layer的参数都是传到对应layer的init处'''
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim, output_dim=hidden1, dropout=self.dropout,
                                              features_nonzero=self.features_nonzero)

        self.hidden2 = Dense(input_dim=self.n_samples, output_dim=hidden1, sparse_inputs=True)

        self.z_u_mean = GraphConvolution(input_dim=hidden1, output_dim=hidden2, dropout=self.dropout)

        self.z_u_log_std = GraphConvolution(input_dim=hidden1, output_dim=hidden2, dropout=self.dropout)

        self.z_a_mean = Dense(input_dim=hidden1, output_dim=hidden2, dropout=self.dropout)

        self.z_a_log_std = Dense(input_dim=hidden1, output_dim=hidden2, dropout=self.dropout)

        self.reconstructions = InnerDecoder(input_dim=hidden2)

    '''模型本质就是两个VAE,一个VAE对x和adj的乘积做encoding生成均值z_u_mean,方差(这里是方差的log)z_u_log_std，做decoding生成preds_sub_u
    另一个对x的转置做生成z_a_mean,z_a_log_std和preds_sub_a'''

    def encode(self, x, adj):
        '''在forward中调用的layer的参数传到对应layer的forward处'''
        z_u = F.relu(self.hidden1(x, adj))
        z_a = torch.tanh(self.hidden2(x.t()))
        return self.z_u_mean(z_u, adj), self.z_u_log_std(z_u, adj), self.z_a_mean(z_a), self.z_a_log_std(z_a)

    def reparameterize(self, z_u_mean, z_u_log_std, z_a_mean, z_a_log_std):
        z_u = z_u_mean + torch.randn_like(z_u_log_std) * torch.exp(z_u_log_std)
        z_a = z_a_mean + torch.randn_like(z_a_log_std) * torch.exp(z_a_log_std)
        return z_u, z_a

    def decode(self, z_u, z_a):
        return self.reconstructions(z_u, z_a)

    def forward(self, features, adj):
        z_u_mean, z_u_log_std, z_a_mean, z_a_log_std = self.encode(features, adj)
        z_u, z_a = self.reparameterize(z_u_mean, z_u_log_std, z_a_mean, z_a_log_std)
        preds_sub_u, preds_sub_a = self.decode(z_u, z_a)
        return preds_sub_u, preds_sub_a, z_u_mean, z_u_log_std, z_a_mean, z_a_log_std
        


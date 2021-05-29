from .model import *
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing

class HOPE(Models):
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
    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut and vt
        ut, s, vt = sp.linalg.svds(matrix, int(dimension / 2))
        emb_matrix_1, emb_matrix_2 = ut, vt.transpose()

        emb_matrix_1 = emb_matrix_1 * np.sqrt(s)
        emb_matrix_2 = emb_matrix_2 * np.sqrt(s)
        emb_matrix_1 = preprocessing.normalize(emb_matrix_1, "l2")
        emb_matrix_2 = preprocessing.normalize(emb_matrix_2, "l2")
        features = np.hstack((emb_matrix_1, emb_matrix_2))
        return features
    def train_model(self, **kwargs):  # (self,rootdir,variable_name,number_walks):
        self.graph=self.mat_content['Network']
        self.G = nx.from_scipy_sparse_matrix(self.graph)
        adj = nx.adjacency_matrix(self.G).todense()
        n = adj.shape[0]
        katz_matrix = np.asarray((np.eye(n) - kwargs['beta'] * np.mat(adj)).I - np.eye(n))
        embeddings = self._get_embedding(katz_matrix, 128)
        return embeddings
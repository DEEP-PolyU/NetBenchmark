import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from .model import *

class Spectral(Models):
    @classmethod
    def is_preprocessing(cls):
        return True

    @classmethod
    def is_deep_model(cls):
        return False

    @classmethod
    def is_end2end(cls):
        return False

    def train_model(self):
        self.dimension = 128
        self.graph = self.mat_content['Network']
        self.G = nx.from_scipy_sparse_matrix(self.graph)
        matrix = nx.normalized_laplacian_matrix(self.G).todense()
        matrix = np.eye(matrix.shape[0]) - np.asarray(matrix)
        ut, s, _ = sp.linalg.svds(matrix, self.dimension)
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        # print(emb_matrix)
        return emb_matrix
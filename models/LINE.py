from .model import *
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing
from tqdm import tqdm

from models.LINE_package.utils import alias_draw, alias_setup

class LINE(Models):
    def check_train_parameters(self):
        space_dtree = {
            'walk_length': hp.uniformint('walk_length', 1, 100),
            'walk_num': hp.uniformint('walk_num', 1, 100),
            'alpha': hp.uniform('alpha', 0, 1),
            'negative': hp.uniformint('negative', 1, 10),
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

    def train_model(self, **kwargs):  # (self,rootdir,variable_name,number_walks):
        self.graph=self.mat_content['Network']
        self.G = nx.from_scipy_sparse_matrix(self.graph)
        self.dimension=128
        self.order=3
        self.negative=int(kwargs['negative'])
        self.init_alpha = (kwargs['alpha'])
        self.is_directed = nx.is_directed(self.G)
        self.num_node = self.G.number_of_nodes()
        self.num_edge = self.G.number_of_edges()
        self.num_sampling_edge = int(kwargs['walk_length'] * int(kwargs['walk_num'] * self.num_node))

        node2id = dict([(node, vid) for vid, node in enumerate(self.G.nodes())])
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]
        self.edges_prob = np.asarray([self.G[u][v].get("weight", 1.0) for u, v in self.G.edges()],'float64')
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        degree_weight = np.asarray([0] * self.num_node)
        for u, v in self.G.edges():
            degree_weight[node2id[u]] += self.G[u][v].get("weight", 1.0)
            if not self.is_directed:
                degree_weight[node2id[v]] += self.G[u][v].get("weight", 1.0)
        self.node_prob = np.power(degree_weight, 0.75)
        self.node_prob /= np.sum(self.node_prob)
        self.node_table, self.node_prob = alias_setup(self.node_prob)

        if self.order == 3:
            self.dimension = int(self.dimension / 2)
        if self.order == 1 or self.order == 3:
            # print("train line with 1-order")
            # print(type(self.dimension))
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self._train_line(order=1)
            embedding1 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 2 or self.order == 3:
            # print("train line with 2-order")
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self.emb_context = self.emb_vertex
            self._train_line(order=2)
            embedding2 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 1:
            self.embeddings = embedding1
        elif self.order == 2:
            self.embeddings = embedding2
        else:
            # print("concatenate two embedding...")
            self.embeddings = np.hstack((embedding1, embedding2))
        return self.embeddings
    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.alpha * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u
    def _train_line(self, order):
        # train Line model with order
        self.alpha = self.init_alpha
        batch_size = 1000
        t0 = time.time()
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = tqdm(range(num_batch))
        for b in epoch_iter:
            if b % 100 == 0:
                # epoch_iter.set_description(
                #     f"Progress: {b *1.0/num_batch * 100:.4f}%, alpha: {self.alpha:.6f}, time: {time.time() - t0:.4f}"
                # )
                self.alpha = self.init_alpha * max((1 - b * 1.0 / num_batch), 0.0001)
            u, v = [0] * batch_size, [0] * batch_size
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5:
                    v[i], u[i] = self.edges[edge_id]

            vec_error = np.zeros((batch_size, self.dimension))
            label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
            for j in range(1 + self.negative):
                if j != 0:
                    label = np.asarray([0 for i in range(batch_size)])
                    for i in range(batch_size):
                        target[i] = alias_draw(self.node_table, self.node_prob)
                if order == 1:
                    self._update(self.emb_vertex[u], self.emb_vertex[target], vec_error, label)
                else:
                    self._update(self.emb_vertex[u], self.emb_context[target], vec_error, label)
            self.emb_vertex[u] += vec_error
import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
from .model import *

class Metapath2vec(Models):
    def check_train_parameters(self):
        space_dtree = {
            'window_size': hp.uniformint('window_size', 1, 20),
            'min-count': hp.uniformint('min-count', 1, 10),
            'walk_length': hp.uniformint('walk_length', 5, 80),
            'walk_num': hp.uniformint('walk_num', 5, 80),
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

    def _walk(self, start_node, walk_length, schema=None):
        # Simulate a random walk starting from start node.
        # Note that metapaths in schema should be like '0-1-0', '0-2-0' or '1-0-2-0-1'.
        if schema:
            schema_items = schema.split("-")
            assert schema_items[0] == schema_items[-1]

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in list(self.G.neighbors(cur)):
                if schema is None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(random.choice(candidates))
            else:
                break
        # print(walk)
        return walk

    def _simulate_walks(self, walk_length, num_walks, schema="No"):
        # Repeatedly simulate random walks from each node with metapath schema.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if schema != "No":
            schema_list = schema.split(",")
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            # print(str(walk_iter + 1), "/", str(num_walks))
            for node in nodes:
                if schema == "No":
                    walks.append(self._walk(node, walk_length))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split("-")[0] == self.node_type[node]:
                            walks.append(self._walk(node, walk_length, schema_iter))
        return walks

    def train_model(self, **kwargs):
        self.schema ='No'
        self.graph=self.mat_content['Network']
        self.dimension = 128
        self.G = nx.from_scipy_sparse_matrix(self.graph)
        self.node_type = self.mat_content['Attributes']
        walks = self._simulate_walks(int(kwargs['walk_length']),int(kwargs['walk_num']), self.schema)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(
            walks,
            size = self.dimension,
            window= int(kwargs['window_size']),
            min_count= int(kwargs['min-count']),
            sg=1,
            workers= 4,
            iter=10,

        )
        id2node = dict([(vid, node) for vid, node in enumerate(self.G.nodes())])
        embeddings = np.asarray([model.wv[str(id2node[i])] for i in range(len(id2node))])
        return embeddings
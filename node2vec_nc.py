# coding: utf-8

# In[15]:

import argparse
import numpy as np
import networkx as nx
import models.node2vec as node2vec
import scipy.io
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count
from evaluation.node_classification import node_classifcation


def read_graph(input_path, directed=False):
    if (input_path.split('.')[-1] == 'edgelist'):
        G = nx.read_edgelist(input_path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())

    elif (input_path.split('.')[-1] == 'mat'):
        edges = list()
        mat = scipy.io.loadmat(input_path)
        nodes = mat['network'].tolil()
        G = nx.DiGraph()
        for start_node, end_nodes in enumerate(nodes.rows, start=0):
            for end_node in end_nodes:
                edges.append((start_node, end_node))

        G.add_edges_from(edges)

    else:
        import sys
        sys.exit('Unsupported input type')

    if not directed:
        G = G.to_undirected()

    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()

    print(nx.info(G) + "\n---------------------------------------\n")
    return G, probs


@node2vec.timer('Generating embeddings')
def generate_embeddings(corpus, dimensions, window_size, num_workers, p, q, input_file, output_file):
    model = Word2Vec(corpus, size=dimensions, window=window_size, min_count=0, sg=1, workers=num_workers)
    # model.wv.most_similar('1')
    w2v_emb = model.wv

    # print('Saved embeddings at : ',output_file)
    # w2v_emb.save_word2vec_format(output_file)

    return model, w2v_emb


def process(input, directed, p, q, d, walks, length, workers, window, output,content):
    Graph, init_probabilities = read_graph(input, directed)
    G = node2vec.Graph(Graph, init_probabilities, p, q, walks, length, workers)
    G.compute_probabilities()
    walks = G.generate_random_walks()
    model, embeddings = generate_embeddings(walks, d, window, workers, p, q, input, output)

    mat = scipy.io.loadmat(input)

    H = np.zeros((mat[content].shape[0], d))
    H[:, 0] = 1
    for nodei in Graph.nodes():
        H[nodei] = embeddings[str(nodei)]
    return H




# def main():
#
#     parser = argparse.ArgumentParser(description = "node2vec implementation")
#
#     parser.add_argument('--input', default='graph/karate.edgelist', help = 'Path for input edgelist')
#
#     parser.add_argument('--output', default=None, help = 'Path for saving output embeddings')
#
#     parser.add_argument('--p', default='1.0', type=float, help = 'Return parameter')
#
#     parser.add_argument('--q', default='1.0', type=float, help = 'In-out parameter')
#
#     parser.add_argument('--walks', default=10, type=int, help = 'Walks per node')
#
#     parser.add_argument('--length', default=80, type=int, help = 'Length of each walk')
#
#     parser.add_argument('--d', default=128, type=int, help = 'Dimension of output embeddings')
#
#     parser.add_argument('--window', default=10, type=int, help = 'Window size for word2vec')
#
#     parser.add_argument('--workers', default=cpu_count(), type=int, help = 'Number of workers to assign for random walk and word2vec')
#
#     parser.add_argument('--directed', dest='directed', action ='store_true', help = 'Specify if graph is directed. Default is undirected')
#     parser.set_defaults(directed=False)
#
#     args = parser.parse_args()
#     process(args)
#
#     return


if __name__ == '__main__':
    embbeding = process(input='../data/POS.mat', directed=False, p=1.0, q=1.0, d=128, walks=4, length=10, workers=12,
                        window=5, output=None,content='network')
    scipy.io.savemat('../node2vec_Embedding.mat', {"node2vec": embbeding})

    mat_contents = scipy.io.loadmat('../data/POS.mat')
    Label = mat_contents["group"]
    matr2 = scipy.io.loadmat('node2vec_Embedding.mat')
    node2vec = matr2['Deepwalk']
    node_classifcation(np.array(node2vec), Label)




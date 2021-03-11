import argparse
import numpy as np
import time
from models.FeatWalk import featwalk
from models.NetMF import netmf
from models.deepwalk import deepwalk
from preprocessing.dataset import Flickr,ACM,Cora,BlogCatalog
from models.Node2vec import node2vec
from models.DGI import DGI
dataAddress = {'Flickr':"data/Flickr/Flickr_SDM.mat"}

datasetlist = [Flickr, ACM, Cora, BlogCatalog]
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}

modellist=[featwalk,netmf,deepwalk,node2vec,DGI]
modeldict = {Cls.__name__.lower(): Cls for Cls in modellist}
def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='cora',choices=datasetdict,
                        help='select a available dataset (default: cora)')
    parser.add_argument('--method', type=str, default='dgi',
                        choices=modeldict,
                        help='The learning method')
    parser.add_argument('--evaluation', type=str, default='node_classification',
                        choices=['node_classification','link_prediction'],
                        help='The evaluation method')
    parser.add_argument('--variable_name', type=str,
                        help='The name of features in dataset')

    args = parser.parse_args()
    return args


def main(args):

    print("Loading...")
    Graph = datasetdict[args.dataset]
    Graph=Graph.get_graph(Graph,variable_name= args.variable_name or 'network' )

    model=modeldict[args.method]
    model(method=args.method, datasets=Graph, evaluation=args.evaluation)





if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
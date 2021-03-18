import argparse
import numpy as np
import time
from models.FeatWalk import featwalk
from models.NetMF import netmf
from models.deepwalk import deepwalk
from preprocessing.dataset import Flickr,ACM,Cora,BlogCatalog
from models.Node2vec import node2vec
from models.DGI import DGI
from models.GAE import GAE
from models.CAN import CAN
from evaluation.node_classification import node_classifcation
import scipy.io as sio
import preprocessing.preprocessing as pre
from evaluation.link_prediction import link_prediction

datasetlist = [Flickr, ACM, Cora, BlogCatalog]
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}

modellist=[featwalk,netmf,deepwalk,node2vec,DGI,GAE,CAN]
modeldict = {Cls.__name__.lower(): Cls for Cls in modellist}
def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='cora',choices=datasetdict,
                        help='select a available dataset (default: cora)')
    parser.add_argument('--method', type=str, default='featwalk',
                        choices=modeldict,
                        help='The learning method')
    parser.add_argument('--evaluation', type=str, default='node_classification',
                        choices=['node_classification','link_prediction'],
                        help='The evaluation method')
    parser.add_argument('--variable_name', type=str,
                        help='The name of features in dataset')
    parser.add_argument('--training_time', type=int, default=200,
                        help='The total training time you want')

    args = parser.parse_args()
    return args



def main(args):

    print("Loading...")
    Graph = datasetdict[args.dataset]
    Graph=Graph.get_graph(Graph,variable_name= args.variable_name or 'network' )
    #iter = get_training_time(args.method,Graph)

    Stoptime = args.training_time
    model=modeldict[args.method]
    model=model(datasets=Graph,iter = iter,Time=Stoptime)

    emb = model.get_emb()
    if args.evaluation == "node_classification":
        node_classifcation(np.array(emb), Graph['Label'])
        sio.savemat('result/' + args.method + '_embedding_'+args.dataset+'.mat', {args.method: emb})
    elif args.evaluation == "link_prediction":
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(Graph['Network'])
        link_prediction(emb, edges_pos=test_edges,edges_neg=test_edges_false)


if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
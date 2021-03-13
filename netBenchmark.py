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
dataAddress = {'Flickr':"data/Flickr/Flickr_SDM.mat"}

datasetlist = [Flickr, ACM, Cora, BlogCatalog]
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}

modellist=[featwalk,netmf,deepwalk,node2vec,DGI,GAE,CAN]
modeldict = {Cls.__name__.lower(): Cls for Cls in modellist}
def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='blogcatalog',choices=datasetdict,
                        help='select a available dataset (default: cora)')
    parser.add_argument('--method', type=str, default='can',
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
    model=model(method=args.method, datasets=Graph, evaluation=args.evaluation)

    if args.evaluation == "node_classification":
        start_time = time.time()
        emb = model.get_emb()
        print("time elapsed: {:.2f}s".format(time.time() - start_time))
        node_classifcation(np.array(emb), Graph['Label'])
        sio.savemat('result/' + args.method + '_embedding_'+args.dataset+'.mat', {args.method: emb})




if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
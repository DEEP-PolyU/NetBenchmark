import argparse
import numpy as np
import time
from models.NetMF import netmf

dataAddress = {'Flickr':"data/Flickr/Flickr_SDM.mat"}

def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='Flickr',choices=['BlogCatalog','ACM','Flickr'],
                        help='select a available dataset (default: Flicker)')
    parser.add_argument('--method', type=str, default='NetMF',
                        choices=['node2vec', 'deepWalk', 'line',
                        'gcn', 'grarep', 'tadw', 'lle', 'hope',
                        'lap', 'gf','sdne','NetMF'],
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
    if args.method == 'NetMF':
        netmf(dataAddress[args.dataset], args.evaluation, variable_name=args.variable_name or'Network' )




if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
import argparse
import numpy as np
import networkx as nx
import time
import os
from preprocessing.dataset import load_adjacency_matrix
from preprocessing.loadCora import load_citation
from models.FeatWalk import featwalk
from models.NetMF import netmf
from models.deepwalk import deepwalk
from preprocessing.dataset import Flickr, ACM, Cora, BlogCatalog, Citeseer, neil001, pubmed, ppi, reddit, yelp, \
    ogbn_arxiv, chameleon, wisconsin, film, squirrel, texas, cornell
from models.Node2vec import node2vec
from models.DGI import DGI
from models.GAE import GAE
from models.GCN import GCN
from models.ProNE import ProNE
from models.CAN_new import CAN_new
from models.CAN_original import CAN_original
from models.GAT import GATModel
from models.HOPE import HOPE
from models.Grarep import Grarep
from models.LINE import LINE
from models.NetSMF import NetSMF
from models.GCN2 import GCN2
from models.SAGE import SAGE
from models.SDNE import SDNE
from evaluation.link_prediction import link_prediction_10_time
from evaluation.node_classification import node_classifcation_10_time
import preprocessing.preprocessing as pre
import copy
from datetime import date

datasetlist = [Cora, Flickr, BlogCatalog, ACM, Citeseer, neil001, pubmed, ppi, ogbn_arxiv, chameleon, wisconsin,
                   film, squirrel]  # yelp,reddit,cornell
datasetlist_all = [Cora, Flickr, BlogCatalog, ACM, Citeseer, neil001, pubmed, ppi, ogbn_arxiv, chameleon, wisconsin,
                   film, squirrel]  # yelp,reddit,cornell
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}
modellist = [featwalk, netmf, deepwalk, node2vec, DGI, GAE, CAN_new, CAN_original, ProNE, HOPE, Grarep, SDNE]
modeldict = {Cls.__name__.lower(): Cls for Cls in modellist}

datasetdict_all = copy.deepcopy(datasetdict)
datasetdict_all['all'] = 1
modeldict_all = copy.deepcopy(modeldict)
modeldict_all['all'] = 1


def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='all', choices=datasetdict_all,
                        help='select a available dataset (default: cora)')
    parser.add_argument('--method', type=str, default='all',
                        choices=modeldict_all,
                        help='The learning method')
    parser.add_argument('--task_method', type=str, default='task1',
                        choices=['task1', 'task2', 'task3'],
                        help='The task method')
    parser.add_argument('--training_time', type=float, default=2.8,
                        help='The total training time you want')
    parser.add_argument('--input_file', type=str, default=None,
                        help='The input datasets you want')
    parser.add_argument('--tuning_method', type=str, default='random',
                        choices=['random', 'tpe'],
                        help='random search/ tpe search')
    parser.add_argument('--cuda_device', type=str, default='0')

    args = parser.parse_args()
    return args

def time_calculating(Graph, training_time_rate):
    edges = list()
    nodes = Graph['Network'].tolil()
    G = nx.DiGraph()
    for start_node, end_nodes in enumerate(nodes.rows, start=0):
        for end_node in end_nodes:
            edges.append((start_node, end_node))

    G.add_edges_from(edges)

    G = G.to_undirected()

    num_of_nodes = G.number_of_nodes()
    num_of_edges = G.number_of_edges()
    time = int(training_time_rate * num_of_nodes)
    print("\n----------Graph infomation-------------\n",
          nx.info(G) + "\n" + "training Time: {}".format(time) + "\n---------------------------------------\n")

    return time


def get_graph_time(args, dkey):
    if (args.input_file == None):
        Graph = datasetdict[dkey]
        Graph = Graph.get_graph(Graph)
        # iter = get_training_time(args.method,Graph)
    Stoptime = time_calculating(Graph, args.training_time)

    return Graph, Stoptime

def main(args):
    today = date.today()
    # deal with the option is not all
    if args.method != 'all':
        temp = modeldict[args.method]
        modeldict.clear()
        modeldict[args.method] = temp
    if args.dataset != 'all':
        temp = datasetdict[args.dataset]
        datasetdict.clear()
        datasetdict[args.dataset] = temp
    # initial variable to store the final result and clean the file
    eval_file_name = 'result/evalFiles/result_' + str(args.tuning_method) + '_' + str(args.method) + '_' + str(
        today) + '_' + str(args.task_method) + '_' + str(args.dataset) + '.txt'
    fileObject = open(eval_file_name, 'w')
    fileObject.close()

    for mkey in modeldict:
        for dkey in datasetdict:
            print("\n----------Train information-------------\n", 'dataset: {} ,Algorithm:{} '.format(dkey, mkey))
            model = modeldict[mkey]
            Graph, Stoptime = get_graph_time(args, dkey)
            model = model(datasets=Graph, iter=iter, time_setting=Stoptime, task_method=args.task_method,
                          tuning=args.tuning_method, cuda=args.cuda_device)
            emb = model.emb
            best = model.best
            tuning_times = model.tuning_times
            temp_result = {'Dataset': dkey, 'model': mkey, 'best': best, 'tuning_times': tuning_times}
            if model.is_end2end():
                f1_mic, f1_mac = model.end2endsocre()
                best = model.get_best()
            else:
                if args.task_method == 'task1' or args.task_method == 'task3':
                    print("node_classification")
                    f1_mic, f1_mac = node_classifcation_10_time(np.array(emb), Graph['Label'])
                    temp_result['f1_micro'] = f1_mic
                    temp_result['f1_macro'] = f1_mac
                elif args.task_method == 'task2':
                    print("link_prediction")
                    roc_score, ap_score = link_prediction_10_time(emb, Graph)
                    temp_result['roc_score'] = roc_score
                    temp_result['ap_score'] = ap_score
                np.save('result/embFiles/' + mkey + '_embedding_' + args.dataset + '.npy', emb)
            # save it in result file by using 'add' model
            fileObject = open(eval_file_name, 'a+')
            fileObject.write(str(temp_result) + '\n')
            fileObject.close()

if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
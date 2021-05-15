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
from preprocessing.dataset import Flickr,ACM,Cora,BlogCatalog
from models.Node2vec import node2vec
from models.DGI import DGI
from models.GAE import GAE
from models.GCN import GCN
from models.CAN_new import CAN_new
from models.CAN_original import CAN_original
from evaluation.evaluation import evaluation
from evaluation.link_prediction import link_prediction
from evaluation.node_classification import node_classifcation
import preprocessing.preprocessing as pre
import copy

datasetlist = [ Cora,Flickr, BlogCatalog,ACM]
datasetdict = {Cls.__name__.lower(): Cls for Cls in datasetlist}
modellist=[featwalk, netmf, deepwalk, node2vec, DGI, GAE, CAN_new, CAN_original]
modeldict = {Cls.__name__.lower(): Cls for Cls in modellist}

datasetdict_all = copy.deepcopy(datasetdict)
datasetdict_all['all'] = 1
modeldict_all = copy.deepcopy(modeldict)
modeldict_all['all'] = 1
def parse_args():
    parser = argparse.ArgumentParser(description='NetBenchmark(DeepLab).')

    parser.add_argument('--dataset', type=str,
                        default='all',choices=datasetdict_all,
                        help='select a available dataset (default: cora)')
    parser.add_argument('--method', type=str, default='all',
                        choices=modeldict_all,
                        help='The learning method')
    parser.add_argument('--evaluation', type=str, default='link_prediction',
                        choices=['node_classification','link_prediction'],
                        help='The evaluation method')
    parser.add_argument('--variable_name', type=str,
                        help='The name of features in dataset')
    parser.add_argument('--training_time', type=int, default=1.4,
                        help='The total training time you want')
    parser.add_argument('--input_file', type=str, default=None,
                        help='The input datasets you want')
    parser.add_argument('--tunning_method', type=str, default='random',
                        help='random search/ tpe search')

    args = parser.parse_args()
    return args

def prase_input_file(args):
    if(os.path.splitext(args.input_file)[-1] == ".mat"):
        Graph = load_adjacency_matrix(dir)
        return Graph
    if(os.path.splitext(args.input_file)[-1] == ".txt"):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str=os.path.splitext(args.input_file)[0])
        Graph = {"Network": adj, "Label": labels, "Attributes": features}
        return Graph
    return None

# TODO(Qian): input file prase

def time_calculating(Graph,training_time_rate):
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
    print("\n----------Graph infomation-------------\n", nx.info(G) +"\n"+ "training Time: {}".format(time) +"\n---------------------------------------\n")


    return time



def get_graph_time(args,dkey):
    if (args.input_file == None):
        Graph = datasetdict[dkey]
        Graph = Graph.get_graph(Graph)
        # iter = get_training_time(args.method,Graph)
    Stoptime = time_calculating(Graph, args.training_time)

    return Graph, Stoptime

def main(args):
    global modeldict


    # prase_input_file(args)

    result_dict = {}
    if args.method !='all':
        temp=modeldict[args.method]
        modeldict.clear()
        modeldict[args.method]=temp
    if args.dataset != 'all':
        temp = datasetdict[args.dataset]
        datasetdict.clear()
        datasetdict[args.dataset] = temp
    i = 0
    for dkey in datasetdict:
        for mkey in modeldict:
            print("\n----------Train information-------------\n",'dataset: {} ,Algorithm:{} '.format(dkey,mkey))
            model = modeldict[mkey]
            Graph,Stoptime = get_graph_time(args,dkey)
            model = model(datasets=Graph, iter=iter, Time=Stoptime,evaluation=args.evaluation,tuning=args.tunning_method)
            emb = model.get_emb()
            best = model.get_best()
            f1_mic, f1_mac = node_classifcation(np.array(emb), Graph['Label'])
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(
                Graph['Network'])
            roc_score, ap_score = link_prediction(emb, edges_pos=test_edges, edges_neg=test_edges_false)

            result_dict[i]= {'Dataset': dkey,'model' : mkey,'f1_micro':f1_mic, 'f1_macro':f1_mac, 'roc_score':roc_score,'ap_score':ap_score,'best':best}
            # print(result_dict[i])
            i += 1
            fileObject = open('result/result.txt', 'w')
            for result in result_dict:
                fileObject.write(str(result_dict[result]))
                fileObject.write('\n')
            fileObject.close()
            # fileObject1 = open('result/data.txt', 'w')
            # fileObject1.close()
            np.save('result/' + mkey + '_embedding_' + args.dataset + '.npy', emb)
    np.save('result.npy', result_dict)

    # result_dict = {}
    # if args.method =='all' and args.dataset == 'all':
    #     i = 0
    #     for dkey in datasetdict:
    #         for mkey in modeldict:
    #             print("\n----------Train information-------------\n",'dataset: {} ,Algorithm:{} '.format(dkey,mkey))
    #             model = modeldict[mkey]
    #             Graph,Stoptime = get_graph_time(args,dkey)
    #             model = model(datasets=Graph, iter=iter, Time=Stoptime,evaluation=args.evaluation,tuning=args.tunning_method)
    #             emb = model.get_emb()
    #             best = model.get_best()
    #             f1_mic, f1_mac = node_classifcation(np.array(emb), Graph['Label'])
    #             adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(
    #                 Graph['Network'])
    #             roc_score, ap_score = link_prediction(emb, edges_pos=test_edges, edges_neg=test_edges_false)
    #
    #             result_dict[i]= {'Dataset': dkey,'model' : mkey,'f1_micro':f1_mic, 'f1_macro':f1_mac, 'roc_score':roc_score,'ap_score':ap_score,'best':best}
    #             # print(result_dict[i])
    #             i += 1
    #             fileObject = open('result/result.txt', 'w')
    #             for result in result_dict:
    #                 fileObject.write(str(result_dict[result]))
    #                 fileObject.write('\n')
    #             fileObject.close()
    #             # fileObject1 = open('result/data.txt', 'w')
    #             # fileObject1.close()
    #             np.save('result/' + mkey + '_embedding_' + args.dataset + '.npy', emb)
    #
    #
    #
    # else:
    #     print("\n----------Train infomation-------------\n", 'dataset: {} ,Algorithm:{} '.format(args.dataset, args.method))
    #     model=modeldict[args.method]
    #     Graph,Stoptime = get_graph_time(args,args.dataset)
    #     model=model(datasets=Graph, iter= iter, Time=Stoptime,evaluation=args.evaluation,tuning=args.tunning_method)
    #     emb = model.get_emb()
    #     value1,value2=evaluation(emb, Graph, args.evaluation)
    #     result_dict[args.method] = [args.dataset,value1, value2]
    #     np.save('result/' + args.method + '_embedding_' + args.dataset + '.npy', emb)





if __name__ == "__main__":
    # np.random.seed(32)
    main(parse_args())
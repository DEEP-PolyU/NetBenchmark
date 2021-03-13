import numpy as np
import scipy.sparse as sp
import scipy.io


def load_AN(dataset):
    edge_file = open(r"data/{}.edge".format(dataset), 'r')
    attri_file = open(r"data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))
        
    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))


    return adj, attribute

def load_AN_mask(dataset,keep_ratio):
    edge_file = open(r"data/{}.edge".format(dataset), 'r')
    attri_file = open(r"data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))

    n, d = attribute.shape
    sparse_mx = attribute.tocoo().astype(np.float32)
    index_row = sparse_mx.row
    index_col = sparse_mx.col

    # randomly mask node attributes for training
    random_index = np.random.random(len(index_row))
    random_index += keep_ratio
    random_index = random_index.astype(int)
    random_index = random_index.astype(np.bool)
    nodeAttriNet = sp.csr_matrix((sparse_mx.data[random_index], (index_row[random_index], index_col[random_index])),
                                 shape=(n, d))
    nodeAttriNet = nodeAttriNet.tocsc()
    print('The origin attribute and the mask attribute is same?: ',
          (attribute.toarray() == nodeAttriNet.toarray()).all())

    return adj, nodeAttriNet

def loadmat(dataset):
    graph=scipy.io.loadmat('data/'+dataset+'.mat')
    if dataset == 'ACM':
        adj=graph['Network']
        attribute=graph['Features']
    else:
        adj = graph['Network']
        attribute=graph['Attributes']

    print('dataset: ',dataset)

    return adj,attribute

def loadmat_knn(dataset,use_net=0,knn=5,ratio=0):
    graph=scipy.io.loadmat('data/'+dataset+'.mat')
    if dataset == 'ACM':
        adj=graph['Network']
        attribute=graph['Features']
    else:
        adj = graph['Network']
        attribute=graph['Attributes']

    print('dataset: ',dataset)

    if type(attribute) is not np.ndarray:
        features = attribute.toarray()
        features[features < 0] = 0.
    if use_net:
        pass
    else:
        if knn == 0:
            if ratio == 0:
                ratio = np.median(features)
            features[features <= ratio] = 0.
        else:
            # feat_index = (-features).argsort()[:knn]
            feat_index = features.argsort()
            feat_index = feat_index[:, -knn:]
            feat_value = np.array([features[i][index] for i, index in enumerate(feat_index)])
            features = np.zeros_like(features)
            for i in range(features.shape[0]):
                indx = feat_index[i]
                features[i, indx] = feat_value[i]
            pass
    features = sp.csr_matrix(features)
    print('---> Total edges for attributes is %d' % features.nnz)
    features=features.tocsc()

    return adj,features

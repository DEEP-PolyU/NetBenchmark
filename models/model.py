import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction,link_prediction_Automatic_tuning
import numpy as np
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial


class Models(torch.nn.Module):
    def __init__(self, *, method, datasets, evaluation, **kwargs):
        super(Models, self).__init__()
        if (self.is_preprocessing == True):
            self.preprocessing(datasets)
        if (self.is_epoch == True):
            self.forward()
        self.mat_content=datasets
        self.method=method
        space_dtree=self.check_train_parameters()
        Label = self.mat_content["Label"]
        trials = Trials()
        algo = partial(tpe.suggest)
        best = fmin(
            fn=self.get_score, space=space_dtree, algo=algo, max_evals=2, trials=trials)
        print(best)
        if evaluation == "node_classification":
            start_time = time.time()
            emb=self.train_model(**best)
            print("time elapsed: {:.2f}s".format(time.time() - start_time))
            node_classifcation(np.array(emb), Label)
            sio.savemat('emb/'+self.method + '_embedding.mat', {self.method: emb})
        if evaluation == "link_prediction":
            matr = sio.loadmat(datasets)
            adj = matr['Network']
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
            link_prediction(emb_name=self.save_emb_name, variable_name=self.model_name, edges_pos=val_edges,
                            edges_neg=val_edges_false)

    def check_train_parameters(self):
        return None

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_epoch(cls):
        raise NotImplementedError

    def forward(self):
        return None

    def train_model(self, **kwargs):
        filename = ""
        return filename

    def get_score(self,params):
        emb = self.train_model(**params)
        adj = self.mat_content['Network']
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_test_edges(adj)
        score=link_prediction_Automatic_tuning(emb,edges_pos=val_edges,edges_neg=val_edges_false)
        return -score

    def preprocessing(self, filename):
        return None

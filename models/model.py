import torch
import scipy.io as sio
import time
import preprocessing.preprocessing as pre
from evaluation.node_classification import node_classifcation
from evaluation.link_prediction import link_prediction,link_prediction_Automatic_tuning
from evaluation.node_classification import node_classifcation_test
import numpy as np
import hyperopt
from hyperopt import fmin, tpe, hp, space_eval,Trials, partial,atpe
import os

class Models(torch.nn.Module):

    def __init__(self, *, datasets, Time, evaluation,tuning,cuda,**kwargs):
        # Train on CPU (hide GPU) due to memory constraints
        # os.environ['CUDA_VISIBLE_DEVICES'] = [0,1,2,3,4,5,6]
        self.use_gpu = torch.cuda.is_available()
        cuda_name = 'cuda:' + cuda
        self.device = torch.device(cuda_name if self.use_gpu else 'cpu')
        self.mat_content=datasets
        self.best = {}
        self.stop_time = Time
        self.evaluation = evaluation
        self.tuning = tuning
        super(Models, self).__init__()
        if self.is_preprocessing():
            self.preprocessing(datasets)
        start_time = time.time()
        if self.is_deep_model():
            emb,best = self.deep_algo()
            self.best = best
            self.emb = emb
            self.end_time = time.time() - start_time
        elif self.is_end2end():
            self.F1_mic, self.F1_mac, self.best = self.end2end()
            self.end_time = time.time() - start_time
        else:
            emb,best = self.shallow_algo()
            self.best = best
            self.emb = emb
            self.end_time = time.time()-start_time


    def check_train_parameters(self):
        return None

    @classmethod
    def is_preprocessing(cls):
        raise NotImplementedError

    @classmethod
    def is_deep_model(cls):
        raise NotImplementedError
    @classmethod
    def is_end2end(cls):
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
        if params['evaluation'] == 'link_prediction':
            score=link_prediction_Automatic_tuning(emb,edges_pos=val_edges,edges_neg=val_edges_false)
        else:
            score=node_classifcation_test(np.array(emb),self.mat_content['Label'])

        return -score
    def en2end_get_score(self,params):
        F1_mic,F1_mac = self.train_model(**params)

        return -F1_mic

    def preprocessing(self, filename):
        return None


    #hyperparameter tuning
    def deep_algo(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning== 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(fn=self.get_score, space=space_dtree, algo=algo, max_evals=1000, trials=trials, timeout=self.stop_time)
        print(best)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**best)

        return emb,best

    def shallow_algo(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning == 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(
            fn=self.get_score, space=space_dtree, algo=algo, max_evals=1000, trials=trials, timeout=self.stop_time)
        print(best)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**best)
        return emb,best

    def end2end(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning == 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(
            fn=self.en2end_get_score, space=space_dtree, algo=algo, max_evals=1000, trials=trials, timeout=self.stop_time)
        print(best)
        print('end of training:{:.2f}s'.format(self.stop_time))
        F1_mic, F1_mac = self.train_model(**best)

        return F1_mic, F1_mac, best

    def end2endsocre(self):
        return self.F1_mic,self.F1_mac

    def get_emb(self):
        return self.emb

    def get_best(self):
        return self.best

    def get_time(self):
        return self.end_time




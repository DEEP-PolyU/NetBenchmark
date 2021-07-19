# NetBenchmark
Node Representation Learning Benchmark


| Status        | Developing      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | QIAN Zhiqiang (zhiqiang.qian@connect.polyu.hk), QUO Zhihao (zhi-hao.guo@connect.polyu.hk), YU Rico (ricoyu.yu@connect.polyu.hk) , LIAN, Amber (amber.lian@connect.polyu.hk) |
| **Updated**   | 2021-05                                           |


## Objective

We aim at building a auto,fair and systematic evaluation platform to compare the results of different Network Embedding models. 
The implemented or modified models include [DeepWalk](https://github.com/phanein/deepwalk),  [node2vec](https://github.com/aditya-grover/node2vec), 
[GCN](https://github.com/tkipf/gcn), [NetMF](https://github.com/xptree/NetMF), GAE, [featWalk](https://github.com/xhuang31/FeatWalk_AAAI19), CAN,LINE,HOPE.

Also, we imported several classic dataset, which includes Flickr, ACM, Cora, BlogCatalog.
We will implement more representative NE models continuously. 
Specifically, we welcome other researchers to contribute NE models into this platform.


## Operation guide 

Download all dependent packages
```bash
    pip install -r requirement.txt
```

Then the command below could be run successfully,The command can select datasets,
algorithms and others through the parameters provided by the command system,which can view details by this command
```bash
    python netBenchmark.py -h
```
optional arguments:

- -h, --help           
  show this help message and exit
-  --dataset {cora,flickr,blogcatalog,citeseer,pubmed,all}      
   select a available dataset (default: all)
-  --method {featwalk,netmf,deepwalk,prone,node2vec,dgi,gae,can_new,can_original,all}         
   The learning method
-  --task_method {task1,task2,task3}       
   The evaluation method
-  --cuda_device {default:0}
   The number of cuda device
-  --training_time TRAINING_TIME (default: 1.4)
   The total training time you want
-  --tuning_method {random,tpe} 
   The method of parameter tuning.(now includes Random search and tpe search)

An example
```bash
CUDA_VISIBLE_DEVICE="0,1,2,3,4,5" python netBenchmark.py --method=all --dataset=all --task_method=task1 --cuda_device=1 
```


## Design detail
### Dataset class: `Dataset`

The dataset component reads in data from ./data, and functions will tackle the original format to a dict result
Until now, we import more than 10 datasets and wrote about 4 kind of reading method.t
All the input datasets inherit from a base class: Dataset.
Path:  ./preprocessing/dataset.py
```python
class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_graph(self,variable_name):
        graph = None
        return graph
    
    @classmethod
    def attributed(cls):
        raise NotImplementedError
```
This class aimed at dealing with a different format of input source files and return a normalized format result. 
We now have 4 different methods to deal with different source files, which includes `mat`/`txt`/`(tx,ty,x,y)`/`npz`.After that, it all will return a DICT result,and adj/labels/featrues will be transformed to csc_matrix
```python
 data={"Network":adj,"Label":labels,"Attributes":feature}
```

### Base class: `Models`
All algorithms model inherit from a base class: ./models/model.py `Models`, which itself inherits from `torch.nn.Module`.
The main idea of this class is to tune parameters and obtain the best result, which will be evaluated by node classification or link prediction according to the different tasks.
So, class `train_model`,`get_score`,`parameter_tuning` are the most important content in model.py.
#### How to import a new algorithm
we build a function  `train_model` for algorithms to normalize the input and output.The dataset processed by Dataset.py will become a global variable that can be called as `self.mat_content`.
And implementation of `train_model`  in different algorithms is various, which means it can be replaced when it comes new algorithms.
`train_model` only had one input parameter called `kwargs`, which will be pre-defined and represent all hyper-parameters of this algorithm.
For example,
```python
 kwargs={'alpha1': 0.2404249370702901, 'num_paths': 47, 'path_length': 48, 'win_size': 14}
```
Then algorithms can call dataset as `self.mat_content` in class and transfer it to embedding.
```python
def train_model(self, **kwargs):
   # need to add the algorithm
   return embedding
```
So, the reason why we used `kwargs` is that for each tuning,the value of it is different,which means it needs to be a variable.
Besides,for each algorithm, its hyper-parameters is varying with numbers,name and range,so it will be recorded in `check_train_parameters`.
Log space is also can be defined here.
For example
```python
 def check_train_parameters(self):
        space_dtree = {
            'alpha1': hp.uniform('alpha1', 0, 1),
            'num_paths': hp.uniformint('num_paths', 10, 50),
            'path_length': hp.uniformint('path_length', 5, 50),
            'win_size': hp.uniformint('win_size', 5, 15),
        }
        return space_dtree
```
All in all, a new algorithms will be imported successfully by overwriting two functions in model.py,which is `train_model` and `check_train_parameters` respectively.
#### Embedding evaluation

In order to tune parameters under different scoring criteria,we wrote get_score function ,which will put the embedding in different evaluation function by an IF-ELSE condition here.
```python
    def get_score(self,params):
        emb = self.train_model(**params)
        adj = self.mat_content['Network']
        if self.task_method == 'task1':
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_val_test_edges(adj)
            score=link_prediction_Automatic_tuning(emb,edges_pos=test_edges,edges_neg=test_edges_false)
        elif self.task_method == 'task2':
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = pre.mask_val_test_edges(adj)
            score = link_prediction_Automatic_tuning(emb, edges_pos=val_edges, edges_neg=val_edges_false)
        else:
            score=node_classifcation_end2end(np.array(emb), self.mat_content['Label'])
        return -score
```
#### Automatic parameter tuning

The third package hyperopt is a reliable package can help us tune parameters.
We choose two methods from it so far,which is random search and TPE respectively.
After tuning and find the best one, it will return the best embbeding and related hyper-parameters we set.
```python
    def parameter_tuning(self):
        trials = Trials()
        if self.tuning == 'random':
            algo = partial(hyperopt.rand.suggest)
        elif self.tuning== 'tpe':
            algo = partial(tpe.suggest)
        else:
            algo = partial(atpe.suggest)

        space_dtree = self.check_train_parameters()
        best = fmin(fn=self.get_score, space=space_dtree, algo=algo, max_evals=1000, trials=trials, timeout=self.stop_time)
        hyperparam = hyperopt.space_eval(space_dtree,best)
        print(hyperparam)
        print('end of training:{:.2f}s'.format(self.stop_time))
        emb = self.train_model(**hyperparam)
        return emb,best
```

### Evaluation class: `Evaluation layer`
Two purpose of evaluation layer is to tune the parameters and obtain the final accuracy fairly.
But deal with embedding will take a lot of time,so, we build 2 function for both link prediction and node classification .
We use five-fold cross-validation to generate the training and test sets.  
In the first round (five rounds in total), we tune the hyper-parameters. We use 1/10 of training set as validationto tune hyperparameters. 
- `node_classifcation_end2end(feature, labels)` 
- `link_prediction_Automatic_tuning(emb, edges_pos, edges_neg)` 
After we have selected the hyperparameters, we add 1/10 validation back to training set. The average of results in ten runs (five rounds in each run) are recorded.
- `node_classifcation_10time(feature, labels)` 10-fold Node classification
- `link_prediction_10_time(emb,Graph)` 


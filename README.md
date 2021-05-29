# NetBenchmark
Node Representation Learning Benchmark


| Status        | Developing      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | QIAN Zhiqiang (zhiqiang.qian@connect.polyu.hk), QUO Zhihao (zhi-hao.guo@connect.polyu.hk), YU Rico (ricoyu.yu@connect.polyu.hk) , LIAN, Amber (amber.lian@connect.polyu.hk) |
| **Updated**   | 2021-05                                           |


## Objective

We aim at building a auto,fair and systematic evaluation platform to compare the results of different Network Embedding models. 
The implemented or modified models include [DeepWalk](https://github.com/phanein/deepwalk),  [node2vec](https://github.com/aditya-grover/node2vec), 
[GCN](https://github.com/tkipf/gcn), [NetMF](https://github.com/xptree/NetMF), GAE, [featWalk](https://github.com/xhuang31/FeatWalk_AAAI19), CAN.

Also, we imported several classic dataset, which includes Flickr, ACM, Cora, BlogCatalog.
We will implement more representative NE models continuously. 
Specifically, we welcome other researchers to contribute NE models into this platform.


## Design overview

### Parser system workflow

The system selects datasets and algorithms through the parameters provided by the command system,which can view details by
```bash
    python netBenchmark.py -h
```
optional arguments:

- -h, --help           
  show this help message and exit
-  --dataset {cora,flickr,blogcatalog,acm,all}      
   select a available dataset (default: cora)
-  --method {featwalk,netmf,deepwalk,node2vec,dgi,gae,can_new,can_original,all}         
   The learning method
-  --evaluation {node_classification,link_prediction}       
   The evaluation method
-  --variable_name VARIABLE_NAME        
   The name of features in dataset
-  --training_time TRAINING_TIME        
   The total training time you want
-  --input_file INPUT_FILE      
   The input datasets you want
-  --tunning_method TUNNING_METHOD      
   The method of parameter tuning.(now includes Random search and tpe search)

An example
```bash
CUDA_VISIBLE_DEVICE="0,1,2,3,4,5" python netBenchmark.py --method=all --dataset=all --cuda_device=1 
```
After choosing the dataset in `datasetdict` and methods in `modeldict`, the parser system will calculate the upper limits of running time and run the following code.

```python
model = modeldict[mkey]
Graph,Stoptime = get_graph_time(args,dkey)
model = model(datasets=Graph, iter=iter, Time=Stoptime,evaluation=args.evaluation,tuning=args.tunning_method,cuda=args.cuda_device)
```

## Design detail
### Dataset class: `Dataset`

All the input datasets inherit from a base class: Dataset.

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
Besides, the input file can be divided into 2 kinds of format,which is the mat file(can be parsed through `scipy.io.loadmat`) and others files(create the special code to parse).

### Base class: `Models`
All algorithms model inherit from a base class: `Models`, which itself inherits from `torch.nn.Module`.
The main idea of this class is to tune parameters and obtain the best result, which will be evaluated by node classification.

```python
class Models(torch.nn.Module):
    
    def __init__(self, *, datasets, time_setting, evaluation,tuning,**kwargs):
       
    def is_end2end(cls):
        raise NotImplementedError

    def check_train_parameters(self)

    def is_preprocessing(cls)
        raise NotImplementedError

    def is_deep_model(cls)
        raise NotImplementedError

    def get_score(self,params)

    def preprocessing(self, filename)

    def hyperparameter_tuning(self)

    def get_emb(self)
        
    def get_best(self)
        
    def get_time(self)
    
    def train_model(self, **kwargs)
```
The following 3 methods should be overridden:

- `is_preprocessing(cls)` Determine whether the model needs to be preprocessed, if it needs to be preprocessed, jump to the preprocessing function for processing

- `is_deep_model(cls)` Determine whether an algorithm is deep model

- `is_deep_model(cls)` Determine whether an algorithm is deep model

- `train_model(self, **kwargs)`Training the datasets and obtain the embedding matrix according to different settings

#### Automatic parameter tuning
Each algorithm's parameters are different, and it will be recorded in `check_train_parameters`.And
```python
trials = Trials()
if self.tuning == 'random':
    algo = partial(hyperopt.rand.suggest)
else:
    algo = partial(tpe.suggest)
space_dtree = self.check_train_parameters()
best = fmin(
    fn=self.get_score, space=space_dtree, algo=algo, max_evals=150, trials=trials, timeout=self.stop_time)
print(best)
print('end of training:{:.2f}s'.format(self.stop_time))
emb = self.train_model(**best)
return emb,best
```

### Evaluation class: `Evaluation layer`

- `node_classifcation(feature, labels)` 10-fold Node classification
- `link_prediction(emb_name,variable_name, edges_pos, edges_neg)`

Examples:
```python
def parse(**kwargs):
    return Evaluation, Graph_Filedir, Model

# parsing
args = {x: y for x, y in args.__dict__.items() if y is not None}
Evaluation, Graph_Filedir, Model = parse(**args)

# train
res = model.train_model(rootdir,**train_args)

# evaluation
if(Evaluation="node_classifcation"):
    node_classifcation()
if(Evaluation="link_prediction"):
    link_prediction()
```
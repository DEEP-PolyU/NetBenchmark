from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters.public_hyper import SPACE_TREE
from .GCN_package.utils import load_data, accuracy, load_citation,load_citationANEmat_gac,load_webANEmat_gac,F1_score
from .GCN_package.input_graph_feed import GraphInput
from .SAGE_package.models import GNN


from .model import *
from preprocessing.preprocessing import load_normalized_format
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
# This is the graphsage version of GCN_package paper

class SAGE(Models):

    @classmethod
    def is_preprocessing(cls):
        return False
    @classmethod
    def is_deep_model(cls):
        return False
    @classmethod
    def is_end2end(cls):
        return True

    def check_train_parameters(self):
        space_dtree = {}
        space_dtree['lr'] = SPACE_TREE['lr']
        space_dtree["nb_epochs"] = SPACE_TREE["nb_epochs"]
        space_dtree["batch_size"] = SPACE_TREE["batch_size"]
        space_dtree["dropout"] = SPACE_TREE["dropout"]
        space_dtree["lamda"] = hp.uniform('lamda', 0, 0.75)
        space_dtree["alpha"] = SPACE_TREE["alpha"]

        return space_dtree

    def train_model(self, **kwargs):

        def train(model, eval_net, data_loader, save, device, train_index, test_index,lr,weight_decay,epoch,step):
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
            data_loader.init_server(train_index=train_index, valid_index=test_index, test_index=test_index)
            max_train_acc = 0.0
            max_valid_acc = 0.0
            min_loss = 1e10
            patience_cnt = 0
            val_loss_values = []
            best_epoch = 0
            t = time.time()
            model.train()
            print('--> Start training...')

            running_loss = 0.0
            steps = step
            epoch_loss = 0.0
            time1 = time.time()

            xx, y = data_loader.next()

            xx = [xx_.to(device) for xx_ in xx]
            y = y.to(device)
            output = model(xx)
            loss_train = F.nll_loss(output, y)
            acc_train = accuracy(output, y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            epoch_loss += loss_train.item()
            # if steps % 100 == 99:
            #     print('--> Epoch %d Step %5d loss: %.3f train_acc: %.3f' % (
            #     epoch + 1, steps + 1, running_loss / 100, acc_train))
            #     running_loss = 0.0
            #
            # eval_net.load_state_dict(model.state_dict())
            # eval_net.eval()
            #
            # train_acc = eval_acc(data_loader, eval_net, device, train_index, test_index, mode='train', )
            # valid_acc = eval_acc(data_loader, eval_net, device,train_index, test_index,mode='test')
            # if train_acc > max_train_acc:
            #     max_train_acc = train_acc
            #     if valid_acc > max_valid_acc:
            #         max_valid_acc = valid_acc
            #         torch.save(model.state_dict(), save)

            print('Epoch: {:04d}'.format(epoch + 1),
                  # 'loss_train: {:.4f}'.format(epoch_loss / steps),
                  # 'acc_train: {:.4f}'.format(train_acc),
                  # 'acc_val: {:.4f}'.format(valid_acc),
                  'time: {:.4f}s'.format(time.time() - time1))

        def eval_acc(loader, net, device, train_index, test_index, mode='test'):
            correct = 0
            total = 0
            data_loader.init_server(train_index=train_index, valid_index=test_index, test_index=test_index)
            with torch.no_grad():
                while True:
                    try:
                        xx, y = loader.next(mode=mode)
                    except StopIteration:
                        break
                    xx = [xx_.to(device) for xx_ in xx]
                    y = y.to(device)
                    outputs = net(xx)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    micro, macro = F1_score(outputs, y)

                acc = correct / total
                # print(micro)
            return micro, macro

        def test(net, test_loader, save, device,train_index ,test_index):
            # net.load_state_dict(torch.load(save))
            net.eval()
            micro, macro = eval_acc(test_loader, net, device,train_index=train_index ,test_index=test_index)
            return micro, macro

        #model start

        semi = 0
        seed = 42
        hidden = '16,16'
        dropout = kwargs["dropout"]
        lr = kwargs["lr"]
        weight_decay = 0
        epochs = int(kwargs["nb_epochs"])
        # epochs = 20
        semi_rate = 1
        kwargs['use_feat'] = 0
        layer_num = '4,10'

        if self.use_gpu:
            device = self.device
            torch.cuda.manual_seed(42)
        else:
            device = self.device
            print("--> No GPU")

        np.random.seed(42)
        torch.manual_seed(42)
        adj, features, labels, _, _, _ = load_normalized_format(datasets=self.mat_content,
                                                                                     semi_rate=semi_rate)

        # Initialize loader
        data_loader = GraphInput(dataset=self.mat_content,layer_num=layer_num,semi=semi_rate)
        u_neighs_num = [str(i) for i in data_loader.u_neighs_num]
        str_neigh = ",".join(u_neighs_num)
        layer_num = str_neigh

        # Model and optimizer
        print('---> Initialize model...')
        model = GNN(nfeat=data_loader.node_dim,
                    nhid=hidden,
                    ndim=layer_num,
                    nclass=data_loader.num_class,
                    dropout=dropout).to(device)
        eval_net = GNN(nfeat=data_loader.node_dim,
                       nhid=hidden,
                       ndim=layer_num,
                       nclass=data_loader.num_class,
                       dropout=dropout).to(device)
        save_path = "models/SAGE_package/savepath/" + 'dataset' + '.pth'

        # test_acc = test(model, data_loader, save_path, device)


        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        t_total = time.time()
        F1_mic_tot = []
        F1_mac_tot = []
        for train_index, test_index in kf.split(features):
            train_index = torch.LongTensor(train_index)
            test_index = torch.LongTensor(test_index)
            train_index.to(device)
            test_index.to(device)
            step = 0
            for epoch in range(epochs):
                train(model, eval_net, data_loader,save_path, device, train_index=train_index, test_index=test_index,lr=lr,weight_decay=weight_decay,epoch=epoch,step=step)
                step += 1
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            # Testing
            F1_mic, F1_mac = test(model, data_loader, save_path, device, train_index=train_index,test_index=test_index)
            F1_mic_tot.append(F1_mic)
            F1_mac_tot.append(F1_mac)
        F1_mic_tot = np.array(F1_mic_tot)
        F1_mac_tot = np.array(F1_mac_tot)
        F1_mic_mean = np.mean(F1_mic_tot)
        F1_mac_mean = np.mean(F1_mac_tot)
        print('F1_mic:', F1_mic_mean)
        print('F1_mac:', F1_mac_mean)

        return F1_mic_mean,F1_mac_mean

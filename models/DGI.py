import torch.nn as nn
import scipy.sparse as sp
from models.dgi_package import GCN, AvgReadout, Discriminator,process
from .model import *
class DGI_test(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_test, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class DGI(Models):
    @classmethod
    def is_preprocessing(cls):
        return False

    @classmethod
    def is_deep_model(cls):
        return True

    def check_train_parameters(self):
        space_dtree = {

            'batch_size': hp.uniformint('batch_size', 1, 100),
            'nb_epochs': hp.uniformint('nb_epochs', 100, 10000),
            'lr': hp.uniform('lr', 0.0001, 0.1), # walk_length,window_size
            'evaluation': str(self.evaluation)
        }

        return space_dtree


    def train_model(self, **kwargs):
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            cuda_name = 'cuda:' + '0'
            device = torch.device(cuda_name)
            torch.cuda.manual_seed(42)
        else:
            device = torch.device("cpu")
            print("--> No GPU")


        # training params
        batch_size = 1
        nb_epochs = int(kwargs["nb_epochs"])
        patience = 20
        lr = kwargs["lr"]
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 128
        sparse = True
        nonlinearity = "prelu"
        adj, features, labels, idx_train, idx_val, idx_test = process.load_citationmat(self.mat_content)
        features, _ = process.preprocess_features(features)
        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = labels.shape[1]

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis])
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        model = DGI_test(ft_size, hid_units, nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available():
            print('Using CUDA')
            model.to(device)
            features = features.to(device)
            if sparse:
                sp_adj = sp_adj.to(device)
            else:
                adj = adj.to(device)
            idx_train = idx_train.to(device)
            idx_val = idx_val.to(device)
            idx_test = idx_test.to(device)
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        start_time = time.time()
        for epoch in range(nb_epochs):
            model.train()
            optimiser.zero_grad()

            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]

            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.to(device)
                lbl = lbl.to(device)

            logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            print('Loss:', loss)

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'models/dgi_package/best_dgi_%d.pkl' % (hid_units))
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break





            loss.backward()
            optimiser.step()

        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('models/dgi_package/best_dgi_%d.pkl' % (hid_units)))

        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)

        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]

        node_emb = embeds.data.cpu().numpy()
        print('node_shape ', node_emb.shape)
        node_emb = node_emb.reshape(node_emb.shape[1:])
        print('node_shape_new ', node_emb.shape)
        return node_emb

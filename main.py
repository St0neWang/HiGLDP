from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from function import *
from model import HiGLDP
from config import Config
import gc
import random
from torch.utils.data import Dataset, DataLoader


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class GraphDataset(Dataset):
    def __init__(self, features, sadj, fadj, asadj, afadj, labels, idx):
        self.features = features
        self.sadj = sadj
        self.fadj = fadj
        self.asadj = asadj
        self.afadj = afadj
        self.labels = labels
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        i = self.idx[index]
        return (self.features[i], self.sadj[i], self.fadj[i], self.asadj[i], self.afadj[i], self.labels[i])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(model, epochs):
    model.train()
    for batch in train_loader:
        features, sadj, fadj, asadj, afadj, labels = batch
        optimizer.zero_grad()
        output = model(features, sadj, fadj, asadj, afadj).cpu()
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    score_auc, score_aupr, score_acc = test(model, test_loader)
    # torch.cuda.empty_cache()
    print("this is the", epochs + 1, "epochs, AUC is {:.4f} and AUPR is {:.4f} accuray is {:.4f},loss is {:.4f} ".format(score_auc, score_aupr, score_acc, loss))

def test(model, test_loader):
    model.eval()
    roc, pr, acc = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features, sadj, fadj, asadj, afadj, labels = batch
            output = model(features, sadj, fadj, asadj, afadj).cpu()
            score_auc, score_aupr, score_acc = RocAndAupr(output, labels)
            roc.append(score_auc)
            pr.append(score_aupr)
            acc.append(score_acc)
    return np.mean(roc), np.mean(pr), np.mean(acc)

if __name__ == "__main__":
    config_file = "./config/200dti.ini"

    config = Config(config_file)
    fold_ROC = []
    fold_AUPR = []


    # K-fold cross-validation
    for fold in range(1, 6):
        config.structgraph_path = config.structgraph_path + str(fold) + '.txt'
        config.train_path = config.train_path + str(fold) + '.txt'
        config.test_path = config.test_path + str(fold) + '.txt'
        use_seed = not config.no_seed
        if use_seed:
            set_random_seed(config.seed)

        sadj, fadj = load_graph(config)
        features, labels, idx_train, idx_test = load_data(config)
        asadj = get_adj(sadj)
        afadj = get_adj(fadj)

        model = HiGLDP(nfeat=config.fdim,
                     nhid1=config.nhid1,
                     nhid2=config.nhid2,
                     nclass=config.class_num,
                     n=config.n,
                     dropout=config.dropout).to(device)

        # features = features.to(device)
        # sadj = sadj.to(device)
        # fadj = fadj.to(device)
        # asadj = asadj.to(device)
        # afadj = afadj.to(device)
        train_dataset = GraphDataset(features, sadj, fadj, asadj, afadj, labels, idx_train)
        test_dataset = GraphDataset(features, sadj, fadj, asadj, afadj, labels, idx_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        roc = []
        pr = []
        acc = []


        acc_max = 0
        epoch_max = 0
        roc_max = 0
        pr_max = 0

        for epoch in range(config.epochs):
            train(model, epoch)
            if acc_max < acc[epoch]:
                acc_max = acc[epoch]
            if roc_max < roc[epoch]:
                roc_max = roc[epoch]
            if pr_max < pr[epoch]:
                pr_max = pr[epoch]
            if epoch + 1 == config.epochs:
                fold_ROC.append(roc_max)
                fold_AUPR.append(pr_max)
                print(
                    "this is {} fold ,the max ROC is {:.4f}, and max AUPR is {:.4f} test set max  accuray is {:.4f} , ".format(
                        fold, roc_max, pr_max, acc_max))

        sadj = sadj.cpu()
        fadj = fadj.cpu()
        features = features.cpu()
        labels = labels.cpu()
        idx_train = idx_train.cpu()
        idx_test = idx_test.cpu()

        del sadj
        del fadj
        del features
        del labels
        del idx_train
        del idx_test

        gc.collect()
        torch.cuda.empty_cache()

    print("average AUROC is {:.4} , average AUPR is {:.4}".format(sum(fold_ROC) / len(fold_ROC),
                                                                  sum(fold_AUPR) / len(fold_AUPR)))

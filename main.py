from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from function import *
from model import HiGLDP
# import tensorflow as tf
from config import Config
# from torch.utils.data import Dataset, DataLoader
import gc
import random


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train(model, epochs):
    model.train()
    optimizer.zero_grad()
    output = model(features, sadj, fadj, asadj, afadj).cpu()
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    score_auc, score_aupr, score_acc = test(model, labels)
    # torch.cuda.empty_cache()
    print("this is the", epochs + 1, "epochs, AUC is {:.4f} and AUPR is {:.4f} accuray is {:.4f},loss is {:.4f} ".format(score_auc, score_aupr, score_acc, loss))

def test(model, labels):
    model.eval()
    with torch.no_grad():
        output = model(features, sadj, fadj, asadj, afadj)
        output = output.cpu()
        labels = labels.cpu()
        score_auc, score_aupr, score_acc = score(output[idx_test], labels[idx_test])
        auc.append(score_auc)
        aupr.append(score_aupr)
        acc.append(score_acc)
    return score_auc, score_aupr, score_acc

if __name__ == "__main__":
    configfile_path = "./config.txt"
    config = Config(configfile_path)
    fold_AUC = []
    fold_AUPR = []

    # K-fold cross-validation
    for fold in range(1, 6):
        config.structgraph_path = './5fold/edge' + str(fold) + '.txt'
        config.train_path = './5fold/train' + str(fold) + '.txt'
        config.test_path = './5fold/test' + str(fold) + '.txt'
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

        features = features.to(device)
        sadj = sadj.to(device)
        fadj = fadj.to(device)
        asadj = asadj.to(device)
        afadj = afadj.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


        auc = []
        aupr = []
        acc = []
        acc_max = 0
        epoch_max = 0
        auc_max = 0
        aupr_max = 0

        for epoch in range(config.epochs):
            train(model, epoch)
            if acc_max < acc[epoch]:
                acc_max = acc[epoch]
            if auc_max < auc[epoch]:
                auc_max = auc[epoch]
            if aupr_max < aupr[epoch]:
                aupr_max = aupr[epoch]
            if epoch + 1 == config.epochs:
                fold_AUC.append(auc_max)
                fold_AUPR.append(aupr_max)
                print(
                    "this is {} fold ,the max AUC is {:.4f}, and max AUPR is {:.4f} test set max  accuray is {:.4f} , ".format(fold, auc_max, aupr_max, acc_max))

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

    print("average AUC is {:.4} , average AUPR is {:.4}".format(sum(fold_AUC) / len(fold_AUC), sum(fold_AUPR) / len(fold_AUPR)))

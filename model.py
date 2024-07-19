import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

import torch
import warnings

warnings.filterwarnings("ignore", message="Converting sparse tensor to CSR format")

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid2)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels=nfeat,
                             out_channels=nhid,
                             heads=4,
                             dropout=dropout)
        self.conv2 = GATConv(in_channels=nhid * 4,
                             out_channels=nhid2,
                             concat=False,
                             heads=1,
                             dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)

        return F.log_softmax(x, dim=1)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # print(z.shape)
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # print(beta.shape)
        return (beta * z).sum(1), beta

class HiGLDP(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(HiGLDP, self).__init__()
        
        # final
        self.SGAT1 = GAT(256, 128, 256, dropout)
        self.SGAT2 = GAT(256, 128, 256, dropout)
        self.CGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN3 = GCN(256, 128, nhid2, dropout)
        self.CGCN4 = GCN(256, 128, nhid2, dropout)
        self.dropout = dropout

        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(2*nhid2)
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(2*nhid2, 64),
            nn.Tanh(),
            nn.Linear(64, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj, asadj, afadj):
        emb1 = torch.relu(self.CGCN1(x, sadj))
        emb2 = torch.relu(self.SGAT1(emb1, asadj))
        emb3 = torch.relu(self.CGCN3(emb2, sadj))
        Xcom = torch.cat((emb1, emb3), dim=1)

        emb4 = torch.relu(self.CGCN2(x, fadj))
        emb5 = torch.relu(self.SGAT2(emb4, afadj))
        emb6 = torch.relu(self.CGCN4(emb5, fadj))
        Ycom = torch.cat((emb4, emb6), dim=1)

        
        emb = torch.stack([Xcom, Ycom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)

        return output






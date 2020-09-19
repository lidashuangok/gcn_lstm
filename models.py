# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
from layer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout,nembed):
        super(GCN,self).__init__()

        self.gc1 = GraphConvolution(nfeat,nhid)
        self.gc2 = GraphConvolution(nhid,nembed)
        self.lstm = nn.LSTM(nfeat, nembed, batch_first=True)
        self.flayer1 = nn.Sequential(nn.Linear(nembed, 8), nn.ReLU(True))
        self.flayer2 = nn.Sequential(nn.Linear(8, nclass))
        self.dropout = dropout

    def forward(self, x,adj,walks):
        x = F.relu(self.gc1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        embeding_l  = self.gc2(x,adj)

        _, (embeds, _) = self.lstm(walks, None)
        embeding_g = embeds[-1]
        embeding = 1*embeding_l+0*embeding_g

        y=self.flayer1(embeding)
        y = self.flayer2(y)
        return F.log_softmax(y, dim=1)

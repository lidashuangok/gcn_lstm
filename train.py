# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
NHID = 16
weight_decay = 5e-4
learning_rate = 0.01
Dropout=0.5
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import graph
from utils import load_data, accuracy
from models import GCN

adj, features, labels, idx_train, idx_val, idx_test,adjn = load_data()

print(features.shape)

model = GCN(features.shape[1],NHID,7,Dropout,32)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
#print(model.parameters())
model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()



G = graph.from_numpy(adjn,undirected=True)

walks_sq = graph.build_deepwalk_corpus(G, num_paths=1,path_length=20, alpha=0, rand=random.Random(0))

#print(walks.__next__())
walks_sq = np.array(walks_sq)
#print(walks.shape)

#inputs = np.empty([2708, 20,1433], dtype = int)
walks = torch.empty([2708, 20,1433], dtype=torch.float)
walks = walks.cuda()
#print(features[walks_sq[0][0]])
for i in range(0,2078):
    for j in range(0,20):
        walks[i][j]= features[walks_sq[i][j]]




def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj,walks)
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    acc_val = accuracy(output[idx_val],labels[idx_val])
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()),'acc_val: {:.4f}'.format(acc_val.item()))

def test():

    model.eval()
    output = model(features, adj,walks)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
t_total = time.time()
for epoch in range(200):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()

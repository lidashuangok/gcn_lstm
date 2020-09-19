from utils import *
import networkx as nx
import random
import graph

import tqdm
import torch
from torch import nn
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
adj, features, labels, idx_train, idx_val, idx_test,adjn = load_data()
print(features.shape)
print(adj.shape,type(adj))
features = features.cuda()
#adj = adj.cuda()
#adjn = adj.cpu().to_dense().numpy()
#featuresa = features.toarray() np.array(a)
print(type(adjn))


# G = nx.from_numpy_matrix(adj.toarray())
# print(len(G))

#all_nodes = list(G.nodes())

G = graph.from_numpy(adjn,undirected=True)

walks = graph.build_deepwalk_corpus(G, num_paths=1,path_length=10, alpha=0, rand=random.Random(0))

#print(walks.__next__())
walks = np.array(walks)
print(walks.shape)

#inputs = np.empty([2708, 20,1433], dtype = int)
inputs = torch.empty([2708, 20,1433], dtype=torch.float)
inputs = inputs.cuda()
print(features[walks[0][0]])
for i in range(0,2078):
    for j in range(0,10):
        inputs[i][j]= features[walks[i][j]]
#inputs = torch.reshape(inputs,[20,2708 ,1433])
print(inputs.shape)


lstm = nn.LSTM(1433,16,batch_first=True).cuda() # input_size, hidden_size, num_layers
#input = torch.randn(5, 3, 10) # time_step, batch, input_size（这里input_size即features）
# h0 = torch.randn(2, 2708, 20).cuda() # num_layers, batch, hidden_size
# c0 = torch.randn(2, 2708, 20).cuda() # num_layers, batch, hidden_size
output, (embeds, cn) = lstm (inputs, None) # output包含从最后一层lstm中输出的ht。shape: time_step, batch, hidden_size

print(embeds.shape)
embeds_sqz = embeds[-1]
print(embeds_sqz.shape)






# inputs_tf = tf.convert_to_tensor(inputs, np.float32)
# lstm = tf.keras.layers.LSTM(16)
# output = lstm(inputs_tf)
# print(type(output))
# print(output.shape)



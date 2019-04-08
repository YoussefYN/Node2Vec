import numpy as np
import pandas as pd
from numpy import genfromtxt

train_path = '/home/ruf/Downloads/train_epin.csv'
IN_path = '/home/ruf/repos/Node2Vec/emb_in.csv'
OUT_path = '/home/ruf/repos/Node2Vec/emb_out.csv'

train = pd.read_csv(train_path).values
IN = genfromtxt(IN_path, delimiter=',')
OUT = genfromtxt(OUT_path, delimiter=',')
K = 10

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

result = []
for i in range(IN.shape[1]):
    node = IN[:,i]
    relevance = node[np.newaxis, :].dot(OUT)
    users_to_drop = np.where(train[:, 0] == i)[0]
    users_to_drop = train[users_to_drop][:,1]
    relevance = relevance.T
    relevance[users_to_drop] = 0
    probabilities = softmax(relevance)
    probabilities[users_to_drop] = 0

    top_K = np.argsort(probabilities, axis=0)#[-K:]
    result += [(i, idx[0]) for idx in top_K]
    break

with open('prediction.csv', mode='w') as f:
    f.write('source_node,destination_node\n')
    for i, idx in result:
        f.write(f'{i},{idx}\n')

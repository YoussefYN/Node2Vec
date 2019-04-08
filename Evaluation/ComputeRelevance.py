import numpy as np
import pandas as pd
from numpy import genfromtxt


train_path = '/home/ruf/Downloads/train_epin.csv'
IN_path = '/home/ruf/repos/Node2Vec/emb_in.csv'
OUT_path = '/home/ruf/repos/Node2Vec/emb_out.csv'

train = pd.read_csv(train_path)
source_nodes = train.get("source_node").values
destination_nodes = train.get("destination_node").value
train = train.values

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

# with open('prediction.csv', mode='w') as f:
#     f.write('source_node,destination_node\n')
#     for i, idx in result:
#         f.write(f'{i},{idx}\n')
N = 40332

def estimate_precision(y_hat):
  MAP = 0.
  for current_node in range(y_hat.keyset()):  # assert len(y_hat.keyset()) == N

    true_destinations = destination_nodes[source_nodes == current_node]  # true destinations for current_node

    GTP = len(true_destinations)  # total number of hidden edges for the node.

    predicted_destinations = y_hat[current_node]  # array for each node in the dict should contain TOP k (10 in our case) elements.

    TP, FP, AP = 0., 0., 0.
    for predicted_destination in predicted_destinations:

      if predicted_destination in true_destinations:
        TP += 1
      else:
        FP += 1
        continue  # AP won't increase if rel (indicator variable) == 0

      current_precision = TP / (TP + FP)

      AP += current_precision / GTP

    MAP += AP

  return MAP / N



import numpy as np
import pandas as pd
from numpy import genfromtxt
import sys

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


def compute_relevance(IN, OUT, train, K=10):
    """
    For each node computes its relevance with all nodes in the dataset,
    except ones that was already seen in the training set. Convert relevance to probabilities.
    Returns top K most relevant nodes.
    """
    recomendations = {}
    for i in range(IN.shape[1]):
        node = IN[:,i]
        relevance = node[np.newaxis, :].dot(OUT)

        users_to_drop = np.where(train[:, 0] == i)[0]
        users_to_drop = train[users_to_drop][:,1]
        relevance = relevance.T
        probabilities = softmax(relevance)
        probabilities[users_to_drop] = 0

        top_K = np.argsort(probabilities, axis=0)[-K:]
        recomendations[i] = top_K.T[0]
    return recomendations



def estimate_precision(y_hat, destination_nodes, N):
    """
    compute Mean Average Precision for recomendations
    """
    MAP = 0.
    for current_node in y_hat.keys():  # assert len(y_hat.keyset()) == N

        true_destinations = destination_nodes[source_nodes == current_node]  # true destinations for current_node
        GTP = len(true_destinations)  # total number of hidden edges for the node.

        predicted_destinations = y_hat[
            current_node]  # array for each node in the dict should contain TOP k (10 in our case) elements.

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



if __name__ == '__main__':
    """
    train_path - path to training data
    IN_path - path to IN matrix
    OUT_path - path to out matrix
    """
    train_path = sys.argv[0]
    IN_path = sys.argv[1]
    OUT_path = sys.argv[2]

    #READING DATA
    train = pd.read_csv(train_path)
    source_nodes = train.get("source_node").values
    destination_nodes = train.get("destination_node").values
    train = train.values
    IN = genfromtxt(IN_path, delimiter=',')
    OUT = genfromtxt(OUT_path, delimiter=',')
    K = 10
    N = IN.shape[1]

    #COMPUTE RECOMENDATION FOR EACH NODE
    recomendations = compute_relevance(IN,OUT,train,K)

    #COMPUTE MEAN AVERAGE PRECISION FOR RECOMENDATIONS
    MAP = estimate_precision(recomendations, destination_nodes, N)
    print(MAP)
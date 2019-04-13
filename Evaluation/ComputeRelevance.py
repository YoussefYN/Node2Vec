import numpy as np
import pandas as pd
import sys
from numpy import genfromtxt


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


def compute_relevance(IN, OUT, train, K=10):
    """
    For each node computes its relevance with all nodes in the dataset,
    except ones that was already seen in the training set. Convert relevance to probabilities.
    Returns top K most relevant nodes.
    """
    N = IN.shape[1]
    recomendations = np.zeros((K, N))
    for i in range(IN.shape[1]):
        node = IN[:, i]
        relevance = node[np.newaxis, :].dot(OUT)
        users_to_drop = np.where(train[:, 0] == i)[0]
        users_to_drop = train[users_to_drop][:, 1]
        relevance = relevance.T
        probabilities = softmax(relevance)
        probabilities[users_to_drop] = -1
        probabilities[i] = -1
        top_K = np.argsort(-probabilities, axis=0)[:K]
        recomendations[:, i] = top_K.reshape(-1, )
    return recomendations

def save_predictions(predictions):
    with open('predictions.csv', mode='w') as f:
        f.write(f'source_node,destination_node\n')
        for i in range(predictions.shape[1]):
            for j in predictions[:,i]:
                f.write(f'{i},{int(j)}\n')
    pass


def estimate_precision(y_hat, destination_nodes, N):
    """
    compute Mean Average Precision for recomendations
    """
    MAP = 0.
    for current_node in range(y_hat.shape[1]):

        # true destinations for current_node
        true_destinations = destination_nodes[source_nodes == current_node]
        GTP = len(true_destinations)  # total number of hidden edges for the node.
        if GTP == 0:
            N -= 1
            continue

        # get top k predictions for current node
        predicted_destinations = y_hat[:, current_node]

        TP, FP, AP = 0., 0., 0.
        for predicted_destination in predicted_destinations:

            if predicted_destination in true_destinations:
                TP += 1
            else:
                FP += 1
                continue  # AP won't increase if rel (indicator variable) == 0

            current_precision = TP / (TP + FP)

            AP += current_precision
        AP /= GTP

        MAP += AP

    return MAP / N

if __name__ == '__main__':
    # PATH TO DATA
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    IN_path = sys.argv[3]
    OUT_path = sys.argv[4]
    # train_path = '/home/ruf/Downloads/train_epin.csv'
    # test_path = '/home/ruf/Downloads/test_epin.csv'
    # IN_path = '/home/ruf/repos/Node2Vec/emb_in.csv'
    # OUT_path = '/home/ruf/repos/Node2Vec/emb_out.csv'

    # READING DATA
    train = pd.read_csv(train_path).values
    test = pd.read_csv(test_path)
    source_nodes = test.get("source_node").values
    destination_nodes = test.get("destination_node").values
    IN = genfromtxt(IN_path, delimiter=',')
    OUT = genfromtxt(OUT_path, delimiter=',')
    K = 10
    N = IN.shape[1]

    # COMPUTE RECOMENDATION FOR EACH NODE
    recomendations = compute_relevance(IN, OUT, train, K)

    #SAVE RECOMENDATIONS
    save_predictions(recomendations)

    # COMPUTE MEAN AVERAGE PRECISION FOR RECOMENDATIONS
    MAP = estimate_precision(recomendations, destination_nodes, N)
    print(MAP)
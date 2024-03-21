import numpy as np



# returns the top1 string
def print_prob(prob, file_path, top_n=1):
    synset = [l.strip() for l in open(file_path).readlines()]

    # Get probabilities
    prob_np = prob.numpy()
    pred = np.argsort(prob_np)[::-1]

    # Get top N labels
    top_n_labels = [(synset[pred[i]], prob_np[pred[i]]) for i in range(top_n)]
    return top_n_labels



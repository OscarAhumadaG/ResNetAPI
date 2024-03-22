import numpy as np



def print_prob(prob, class_names, top_n=1):
    # Get top N labels
    top_n_labels = [(class_names[i], prob[i]) for i in range(top_n)]
    return top_n_labels



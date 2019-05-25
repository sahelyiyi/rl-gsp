import random
import numpy as np


def calculate_labels(graph, M):  # TODO CORRECT IT
    labels = []
    for i in range(len(graph.nodes)):
        labels.append(random.randint(0, len(graph.nodes)))
    return np.array(labels)

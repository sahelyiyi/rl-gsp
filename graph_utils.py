import networkx as nx
import numpy as np

from collections import defaultdict
from random import randint

from settings import MIN_CLUSTERS_CNT, MAX_CLUSTERS_CNT, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE, P_IN, P_OUT


def generate_random_graph():
    clusters_cnt = randint(MIN_CLUSTERS_CNT, MAX_CLUSTERS_CNT)
    clutsers_data = []
    nodes_cnt = 0
    for i in range(clusters_cnt):
        cluster_size = randint(MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE)
        nodes_cnt += cluster_size
        clutsers_data.append(cluster_size)

    while True:
        graph = nx.random_partition_graph(clutsers_data, P_IN, P_OUT)
        if nx.is_connected(graph):
            break

    x_original = [None] * nodes_cnt
    for cluster_num, cluster in enumerate(graph.graph['partition']):
        for node in cluster:
            x_original[node] = cluster_num

    # nx.draw(graph)
    # plt.show()

    return graph, np.array(x_original)


def get_neighbors_data(graph):
    neighbors_data = defaultdict(lambda: defaultdict(list))
    for node1, node_data in nx.all_pairs_shortest_path_length(graph):
        for node2 in node_data:
            distance = node_data[node2]
            neighbors_data[node1][distance].append(node2)
    return neighbors_data

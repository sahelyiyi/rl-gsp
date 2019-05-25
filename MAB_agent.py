from random import choice

import numpy as np

from graph_utils import get_neighbors_data
from labeling import calculate_labels
from settings import NEXT_NODE_MAX_ITER


class Agent:
    def __init__(self, H, batch_size, alpha, max_episodes, gamma):

        self.H = H
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_episodes = max_episodes
        self.gamma = gamma

        self.W = np.zeros(shape=(self.H, 1)).flatten()
        self.policy = np.full(shape=(self.H, 1), fill_value=1.0/self.H).flatten()
        self.delta_w = np.zeros(shape=(self.H, 1)).flatten()
        self.g = np.zeros(shape=(self.H, 1)).flatten()
        self.rewards = []

    @staticmethod
    def sample_node(neighbors, M):
        if not neighbors:
            return None

        next_node = choice(neighbors)
        cnt = 1
        while next_node in M:
            next_node = choice(neighbors)
            cnt += 1
            if cnt == NEXT_NODE_MAX_ITER:
                return None

        return next_node

    def sample_action(self):
        return np.random.choice(np.arange(0, len(self.policy)), p=self.policy)

    def sampling(self, graph, neighbors_data, sample_size):
        node = choice(list(graph.nodes()))
        M = set([node])
        L = []
        next_node = None
        for t in range(1, sample_size):
            while not next_node:  # TODO FIX INFINIT LOOP
                a = self.sample_action()
                next_node = self.sample_node(neighbors_data[node][a + 1], M)
            M.add(next_node)
            L.append(a)
            node = next_node
            next_node = None

        return M, L

    @staticmethod
    def calculate_R(x_original, x):
        return -1.0 * np.square(x_original - x).mean()

    def pull(self, graph, x_original, neighbors_data, sample_size):
        M, L = self.sampling(graph, neighbors_data, sample_size)

        x = calculate_labels(graph, M)
        R = self.calculate_R(x_original, x)
        self.rewards.append(R)

        for k in range(sample_size-1):
            for a in range(self.H):
                if a == L[k]:
                    self.delta_w[a] = self.delta_w[a] + R * (1 - self.policy[a])
                else:
                    self.delta_w[a] = self.delta_w[a] - R * self.policy[a]

    def learn(self, graph, x_original, sample_size):
        neighbors_data = get_neighbors_data(graph)

        for episode_num in range(self.max_episodes):

            self.pull(graph, x_original, neighbors_data, sample_size)

            if episode_num != 0 and episode_num % self.batch_size == 0:
                self.g = self.gamma * self.g + (1 - self.gamma) * np.power(self.delta_w, 2)
                self.W = self.W + self.alpha * self.delta_w / np.sqrt(self.g)
                self.policy = (np.exp(self.W) / np.sum(np.exp(self.W))).flatten()
                self.policy = self.policy / np.sum(self.policy)
                self.delta_w = np.zeros(shape=(self.H, 1)).flatten()

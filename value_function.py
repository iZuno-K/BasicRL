import numpy as np
import random


class Q_learning(object):
    def __init__(self, state_range, action_dim, gamma=0.9, alpha=0.3, epsilon=0.03):
        self.q_table = np.zeros(tuple(state_range) + (action_dim,))  # I assume state_range is 2d array like (3, 4)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.a_dim = action_dim

    def update(self, trajectory):
        # reverse order
        for traj in trajectory[::-1]:
            s0 = traj[0]
            a = traj[1]
            r = traj[2]
            s1 = traj[3]
            self.q_table[s0[0], s0[1], a] = (1. - self.alpha) * self.q_table[s0[0], s0[1], a] + self.alpha * (
                        r + self.gamma * np.max(self.q_table[s1[0], s1[1]]))

    def optimal_action(self, state):
        max_q = None
        s = state
        for i, q in enumerate(self.q_table[s[0], s[1]]):
            if max_q is None:
                max_q = q
            else:
                if max_q < q:
                    max_q = q

        if max_q is None:
            return random.randint(0, self.a_dim - 1)
        else:
            # consider multiple maximums
            candidate = np.where(self.q_table[s[0], s[1]] == max_q)[0]
            l = len(candidate)
            idx = random.randint(0, l-1)
            return candidate[idx]

    def act(self, state, exploration=True):
        if exploration:
            if np.random.uniform(0, 1) <= self.epsilon:
                return random.randint(0, self.a_dim - 1)
            else:
                return self.optimal_action(state)
        else:
            a = self.optimal_action(state)
            return a

    def save_table(self, save_path):
        np.save(save_path, self.q_table)

    def load_table(self, load_path):
        self.q_table = np.load(load_path)


class ValueFunction(object):
    """
    Debugging ...
    """
    def __init__(self, state_range, gamma=0.99, alpha=0.3):
        self.v_table = np.zeros(state_range)
        self.gamma = gamma
        self.alpha = alpha

    def update(self, trajectory):
        # reverse order
        for i, traj in enumerate(trajectory[::-1]):
            s0 = traj[0]
            # a = traj[1]
            r = traj[2]
            s1 = traj[3]
            # if i == 0:
            #     self.v_table[s1[0], s1[1]] = (1. - self.alpha) * self.v_table[s1[0], s1[1]] + self.alpha * r
            self.v_table[s0[0], s0[1]] = (1. - self.alpha) * self.v_table[s0[0], s0[1]] + self.alpha * (
                        r + self.gamma * self.v_table[s1[0], s1[1]])

    def save_table(self, save_path):
        np.save(save_path, self.v_table)

    def load_table(self, load_path):
        self.v_table = np.load(load_path)
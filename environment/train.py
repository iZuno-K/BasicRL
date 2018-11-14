import sys
from pathlib import Path
print(Path(__file__).parent)
sys.path.append(Path(__file__).parent)

sys.path.append()
from environment.discrete_maze import Discrete2DMaze
from value_function import Q_learning
import numpy as np


def train():
    env = Discrete2DMaze()
    obs = env.reset()
    size = env.maze_size
    q = Q_learning(state_range=size, action_dim=4, epsilon=0.1)
    print("Training")
    for i in range(100):
        print("episode:{}".format(i))
        obs = env.reset()
        done = False
        trajectory = []
        while not done:
            action = q.act(state=obs, exploration=True)
            next_obs, rew, done, info = env.step(action)
            trajectory.append([obs, action, rew, next_obs])
            obs = next_obs
            env.render(value_map=np.max(q.q_table, axis=2))
            print(rew)
        q.update(trajectory)

    print("Exploitation")
    for i in range(10):
        print("episode:{}".format(i))
        obs = env.reset()
        done = False
        while not done:
            action = q.act(state=obs, exploration=True)
            next_obs, rew, done, info = env.step(action)


if __name__ == '__main__':
    train()

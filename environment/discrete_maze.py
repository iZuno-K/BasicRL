from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Discrete2DMaze(object):
    """
    [1, 0, 0] indicates there is a key
    [0, 1, 0] indicates there is a door
    [0, 0, 1] indicates there is nothing
    [0, 0, 0] indicates there is wall (agent cannot move there)
    array order corresponds to  [y][x]
    """

    def __init__(self):
        self.env_id = "DiscreteMaze"
        self.maze_size = np.array([5, 5])

        self.map = np.zeros(self.maze_size, dtype=np.int64)
        self.unreachable = []  # 壁の設定など

        self.init_position = np.zeros(2, dtype=np.int64)
        self.agent_position = self.init_position

        self.action_decoder = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.door_pos = self.maze_size - 1
        obs = self.reset()
        self.observation_shape = obs.shape
        self.action_range = 6

    def step(self, action):
        """
        :param int action: 0, 1, 2, 3
        :return:
        """
        info = {}
        if action < 4:
            next_pos = self.agent_position + self.action_decoder[action]

            for pos in self.unreachable:
                if (self.agent_position == pos).all():
                    next_pos = self.agent_position   # 壁とかいけないとこにはいけないように
                    break
            self.agent_position = next_pos
            self.agent_position = np.clip(self.agent_position, np.array([0, 0]), self.maze_size-1)  # はみ出ないように
        else:
            raise AssertionError("Action range is 0-3 but receive {}".format(action))

        if (self.agent_position == self.door_pos).all():
            done = True
            reward = 1.
        else:
            done = False
            reward = 0.

        obs = self.make_observation()

        return obs, reward, done, info

    def reset(self):
        self.agent_position = self.init_position
        return self.make_observation()

    def make_observation(self):
        return self.agent_position

    def render(self, value_map=None):
        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()
            path = str(Path(__file__).parent)
            path += '/imgs/'
            agent = OffsetImage(plt.imread(path + 'agent.png'), zoom=0.12)
            door = OffsetImage(plt.imread(path + 'door.png'), zoom=0.12)
            self.imgs = [agent, door]

            # plot grid worlds
            plt.yticks(np.arange(0.5, self.maze_size[0], 1))
            plt.xticks(np.arange(0.5, self.maze_size[1], 1))
            self.ax.tick_params(labelbottom=False, bottom=False)
            self.ax.tick_params(labelleft=False, left=False)

            maze = np.zeros(self.maze_size, dtype=np.int64)
            self.im = plt.imshow(maze, cmap="Reds",  vmin=0., vmax=1.,)
            self.ax.grid(color='k', linestyle='-', linewidth=2)

        if hasattr(self, "tmp"):
            [ar.remove() for ar in self.tmp]

        if value_map is not None:
            maze = self.normalize(value_map)
            self.im.set_array(maze)
        a = AnnotationBbox(self.imgs[0], self.agent_position[::-1], xybox=(0, 0), xycoords="data",
                           boxcoords="offset points")
        self.ax.add_artist(a)
        self.tmp = [a]
        if self.door_pos is not None:
            b = AnnotationBbox(self.imgs[1], self.door_pos[::-1], xycoords="data", boxcoords="offset points")
            self.ax.add_artist(b)
            self.tmp.append(b)
        # plt.show(block=False)
        plt.pause(.01)

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    @staticmethod
    def normalize(arr):
        m = np.min(arr)
        arr = arr - m
        M = np.max(arr)
        if M != 0:
            arr = arr / M
        return arr

if __name__ == '__main__':
    # env = Discrete2DMaze()
    env = Discrete2DMaze()
    obs = env.reset()
    for i in range(100):
        _obs, rew, _done, _info = env.step(np.random.randint(0, 3))
        print(_obs)
        env.render()
        if _done:
            env.reset()
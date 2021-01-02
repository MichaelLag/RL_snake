import numpy as np
import matplotlib.pyplot as plt
import random


class my_snake:
    def __init__(self, map_size=10):
        self.life = 0
        self.L = map_size * map_size
        self.state_size = self.L
        self.rand_generator = np.random.RandomState(4)
        self.map_size = map_size
        self.full_state = np.zeros((map_size, map_size), dtype=int)
        edge1 = np.vstack((np.arange(0, map_size), np.zeros((1, map_size))))
        edge2 = np.vstack((np.arange(0, map_size), np.zeros((1, map_size)) + map_size - 1))
        edge3 = np.vstack((np.zeros((1, map_size)), np.arange(0, map_size)))
        edge4 = np.vstack((np.zeros((1, map_size)) + map_size - 1, np.arange(0, map_size).T))
        borders = np.concatenate((edge1, edge2, edge3, edge4), axis=1).T.astype(int)
        self.terminal = 0
        self.target_coord = []
        self.snake_state = np.zeros((map_size, map_size))
        self.borders = np.zeros((map_size, map_size))
        self.target = np.zeros((self.map_size, self.map_size))
        self.borders[borders[:, 0], borders[:, 1]] = 1
        self.snake_state_dict = {new_list: [] for new_list in range(self.L)}
        self.snake_state_dict[self.L] = np.array([map_size - 2, 3])
        self.snake_state_dict[self.L - 1] = np.array([map_size - 2, 2])
        self.snake_state_dict[self.L - 2] = np.array([map_size - 2, 1])
        self.map_state_dict()
        self.ax = None
        self.action_size = 3
        self.H = np.zeros((map_size, map_size))
        self.evaluate_state()
        self.evaluate_target()
        self.evaluate_state()
        self.reward = 0

    def print_state(self):
        print(self.H)
    def reward(self):
        return self.reward
    def state_flat(self):
        return self.H.reshape((1,-1)).squeeze()

    def state_flat_long(self):
        short_H = self.H[1:-1,1:-1]
        snake   = np.array([short_H == 2]).reshape((1, -1)).squeeze()
        head    = np.array([short_H == 3]).reshape((1, -1)).squeeze()
        target  = np.array([short_H == 4]).reshape((1, -1)).squeeze()
        return np.concatenate((snake, head, target), axis=0).astype(int)

    def evaluate_target(self):
        self.target *= 0
        while True:
            x = random.choice(np.arange(1, self.map_size))
            y = random.choice(np.arange(self.map_size - 1, 0, -1))
            if self.H[x, y] == 0:
                self.target[x, y] = 4
                self.target_coord = [x,y]
                break

    def evaluate_state(self):
        self.map_state_dict()
        H = self.borders + self.snake_state + self.target
        self.H = H

    def disp_state(self):
        plt.ion()
        if self.ax == None:
            fig = plt.figure(figsize=(16, 16))
            ax = fig.add_subplot(111)
            ax.set_title('Snake Current State')
            ax.set_aspect('equal')
            # Major ticks
            ax.set_xticks(np.arange(0, self.map_size, 1))
            ax.set_yticks(np.arange(0, self.map_size, 1))
            # Minor ticks
            ax.set_xticks(np.arange(-.5, self.map_size, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.map_size, 1), minor=True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            # Gridlines based on minor ticks
            ax.grid(which='minor', color='white', linestyle='-', linewidth=3)
            self.ax = ax
            self.fig = fig

        H = self.borders + self.snake_state + self.target
        print(H)
        self.ax.imshow(H, cmap='hot', vmin=0, vmax=4)
        plt.show()
        plt.pause(3)
        action = input('action?\n')
        print('Got action: {}'.format(action))

    def move_step(self, action):
        # 0 - do nothing
        # 1 - move left
        # 2 - move right
        self.perform_step(action)
        if np.sum(self.H==4)>1 or not np.any(self.H==3):
            self.terminal = 1
            self.reward   = -100
        return self.life, self.terminal, self.reward

    def map_state_dict(self):
        self.snake_state = np.zeros((self.map_size, self.map_size))

        for key in np.arange(self.L, self.L - self.life - 3, -1):
            # 3 is initial length
            x = self.snake_state_dict[key]
            if x == []:
                break
            if key == self.L:
                self.snake_state[x[0], x[1]] = 3
            else:
                self.snake_state[x[0], x[1]] = 2

    def perform_step(self, action):
        head0 = self.snake_state_dict[self.L]
        head1 = self.snake_state_dict[self.L - 1]
        dR0 = np.array(head0) - np.array(head1)
        dRT = -np.array(head0) + np.array(self.target_coord)

        if action == 0:
            R = np.array([[1, 0], [0, 1]])
        elif action == 1:
            R = np.array([[0, -1], [1, 0]])
        else:
            R = np.array([[0, 1], [-1, 0]])

        for key in np.arange(self.L - self.life - 2,self.L):
            self.snake_state_dict[key] = self.snake_state_dict[key+1].copy()

        self.snake_state_dict[self.L] += np.matmul(R, dR0)
        self.evaluate_state()
        self.reward = -2 * (np.dot(dRT,dR0)<=0) + 1* (np.dot(dRT,dR0)>0)

        if self.H[self.snake_state_dict[self.L][0],self.snake_state_dict[self.L][1]] == 7:
            self.life += 1
            self.reward = 100#30
            self.snake_state_dict[self.L - self.life - 4] = self.snake_state_dict[self.L - self.life - 3].copy()
            self.evaluate_state()
            self.evaluate_target()
            self.evaluate_state()





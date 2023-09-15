import random, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy.io import loadmat

class World:
    def __init__(self, x_lower, y_lower, x_upper, y_upper, obs=16):
        self.origincorner = [x_lower, y_lower]    # [0, 0]
        self.endcorner = [x_upper, y_upper]   # [20, 20]
        self.num_obs = obs   #  random.randint(16, 16)
        self.obs = []
        self.scale = 10
        self.obs_map = np.zeros((x_upper * self.scale + 1, y_upper * self.scale + 1))

    def obstacles(self):
        self.obs = []
        self.obs_map = self.obs_map * 0
        for k in range(0, self.num_obs):
            x = (self.endcorner[0] - 0.5) * random.random() + 0.1
            y = (self.endcorner[1] - 0.5) * random.random() + 0.1
            a = 0.5 + 2 * random.random()
            b = 0.5 + 2 * random.random()
            while (x + a) >= self.endcorner[0]:
                x = (self.endcorner[0] - 0.5) * random.random() + 0.1
                a = 0.5 + 2 * random.random()
            while (y + b) >= self.endcorner[1]:
                y = (self.endcorner[1] - 0.5) * random.random() + 0.1
                b = 0.5 + 2 * random.random()
            self.obs.append([x, y, x + a, y + b])
            for i in range(math.floor(x*self.scale), math.ceil((x+a)*self.scale)):
                for j in range(math.floor(y*self.scale), math.ceil((y+b)*self.scale)):
                    self.obs_map[i][j] = 1

    def plt_world(self):
        fig, ax = plt.subplots()
        ax.plot([0, self.endcorner[0], self.endcorner[0], 0, 0], [0, 0, self.endcorner[1], self.endcorner[1], 0])
        for i in range(self.num_obs):
            ax.plot([self.obs[i][0], self.obs[i][2], self.obs[i][2], self.obs[i][0], self.obs[i][0]],
                     [self.obs[i][1], self.obs[i][1], self.obs[i][3], self.obs[i][3], self.obs[i][1]], color='r')
            ax.add_patch(Rectangle((self.obs[i][0],self.obs[i][1]), self.obs[i][2]-self.obs[i][0], self.obs[i][3]-self.obs[i][1]))

    def check_point(self, p):
        p = p * self.scale
        flag = 0
        if self.obs_map[math.floor(p[0])][math.floor(p[1])]:
            flag = 1
            return flag
        return flag

    def check_path(self, traj):
        flag = 0
        traj = traj * self.scale
        for k in range(np.size(traj, 0)):
            p = traj[k]
            if self.obs_map[math.floor(p[0])][math.floor(p[1])]:
                flag = 1
                return flag
        return flag


if __name__ == '__main__':
    world = World(0, 0, 20, 20, 8)
    '''
    graphdata = loadmat('GraphPY1.mat')
    Vertices = graphdata['Vertices']
    world_obs = graphdata['world_obs']
    world_obs_map = graphdata['world_obs_map']    
    world.obs_map = world_obs_map
    world.obs = world_obs
    '''
    world.obstacles()
    world.plt_world()
    plt.show()
    '''
    output = open('Plans.pkl', 'wb')
    pickle.dump(world, output)
    output.close()
    pkl_file = open('Plans.pkl', 'rb')
    world = pickle.load(pkl_file)
    pkl_file.close()
    '''
    '''
    print(world.obs)
    print(world.num_obs)
    world.plt_world()

    path = np.array([[2 + i * 16 / 100, 1 + i * 18 / 100] for i in range(100)])
    flag = world.check_path(path)
    print(flag)
    path = np.transpose(path)
    plt.plot(path[0], path[1])

    path = np.array([[2 + i * 6 / 100, 1 + i * 3 / 100] for i in range(100)])
    flag = world.check_path(path)
    print(flag)
    path = np.transpose(path)
    plt.plot(path[0], path[1])
    plt.show()
    '''
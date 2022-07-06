import numpy as np
import random

LEFT = 1
DOWN = 2
RIGHT = 3
UP = 4

MAPS = {
    "10x10": [
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000"
    ],
    "5x5": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000"
    ]
}
# Agent class
class Agent():
    def __init__(self,type,id):
        self.id = id
        self.type = type
        if type=='predator':
            self.nA = 5
            self.action_space = [0,1,2,3,4]
        else:
            self.nA = 9
            self.action_space = [0,1,2,3,4,5,6,7,8]
        self.state = (0,0)
        self.delayed_state = None

    def set_state(self,state):
        self.state = state

    def set_delayed_state(self,ds):
        self.delayed_state = ds

# Environment class
class env():
    def __init__(self, map_name="10x10", nagents=2, npreys=1,reward=0):
        desc = MAPS[map_name]
        self.reward=reward
        self.size = int(map_name.split('x')[0])
        self.nagents = nagents
        self.npreys = npreys
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.agents = [Agent('predator',i) for i in range(nagents)]
        self.preys = [Agent('prey',i) for i in range(npreys)]
        [a.set_state(tuple(np.random.randint(0, high=self.size, size=2))) for a in self.agents+self.preys]
        self.nA = 5
        self.nS = ((nrow * ncol) ** npreys)* ((nrow * ncol)**nagents)
        self.state_shape = [self.size for i in range(2*(nagents+npreys))]
        self.map = np.array(desc == b'0').astype('float64').ravel()
        self.v = [self.size**((self.nagents+self.npreys)*2 -1-i) for i in range((self.nagents+self.npreys)*2)]
        self.state_action_space = tuple([self.size for i in range(2*nagents)]+\
                                  [self.size for i in range(2*npreys)]+\
                                  [self.nA for i in range(nagents)])

    def to_s(self, s):
        return int(np.dot(s,self.v))

    def global_state(self):
        return np.concatenate([a.state for a in self.agents+self.preys])

    def agent_state(self):
        return np.concatenate([a.state for a in self.agents])

    def prey_state(self):
        return np.concatenate([a.state for a in self.preys])

    def to_a(self,a):
        return [int(a/self.nA), int(a%self.nA)]

    def inc(self, row, col, a):
        if a == 1:
            col = max(col - 1, 0)
        elif a == 2:
            row = min(row + 1, self.nrow - 1)
        elif a == 3:
            col = min(col + 1, self.ncol - 1)
        elif a == 4:
            row = max(row - 1, 0)
        elif a == 5:
            row = min(row + 1, self.nrow - 1)
            col = min(col + 1, self.ncol - 1)
        elif a == 6:
            row = max(row - 1, 0)
            col = min(col + 1, self.ncol - 1)
        elif a == 7:
            row = max(row - 1, 0)
            col = max(col - 1, 0)
        elif a == 8:
            row = min(row + 1, self.nrow - 1)
            col = max(col - 1, 0)
        return (row, col)

    def step(self,action):
        oldstate = []
        newstate = []
        oldpreys = []
        newpreys = []
        collision = False
        done = False
        for u,a in zip(action,self.agents):
            oldstate.append(a.state)
            newstate.append(self.inc(oldstate[-1][0],oldstate[-1][1],u))
            a.set_state(newstate[-1])
        if len(set(newstate))==1:
            collision = True
        #random prey move
        for a in self.preys:
            si = a.state
            oldpreys.append(si)
            newpreys.append(self.inc(si[0], si[1], random.choice([0,1,2,3,4,5,6,7,8])))
            a.set_state(newpreys[-1])
        if collision:
            if newstate[0] in oldpreys:
                reward = 1
                done = True
            else:
                reward = -1
        else:
            reward = self.reward
        obs = np.concatenate(newstate + newpreys)
        return obs,reward,done,None

    def reset(self):
        [a.set_state(tuple(np.random.randint(0, high=self.size, size=2))) for a in self.agents+self.preys]



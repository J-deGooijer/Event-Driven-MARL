import random
import numpy as np
from collections import deque
from particle_tag import ParticleTag

# Main simulator class.
class MultiAgentParticle:
    def __init__(self,gamma,alpha,Nagents,map_name='10x10',reward=0):
        mname = map_name
        self.env = ParticleTag.env(map_name=mname, nagents=Nagents, npreys=1,reward=reward)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 0.2
        self.updates = []
        self.losses = []
        self.neighbours = {}
        self.s = int(mname.split('x')[0])
        self.q = np.random.random_sample(self.env.state_action_space)
        self.q = None
        self.qA = None
        self.qB = None
        self.B = deque([], maxlen=Nagents*100000)
        self.cr = 0
        self.delta = None
        self.transmissions = 0
        self.e = 0

    def load_e(self,e):
        self.e = e

    def reset(self):
        self.env.reset()
        self.last_states = {}

    def load_Q(self,Q):
        self.q = Q

    def load_B(self,B):
        self.B = B

    def reset_epsilon(self):
        self.epsilon = 1

    def load_delta(self,delta):
        self.delta = delta

    def policy(self,qvalues,epsilon=1):
        action = self.epsilon_greedy_q_policy(qvalues,epsilon=epsilon)
        return action

    def to_s(self, s):
        v = [self.s**5,self.s**4,self.s**3,self.s**2,self.s**1,self.s**0]
        return int(np.dot(s,v))

    def gamma_eval(self, x): #Computes the point value for a Gamma function. Treats it as SVR unless exception.
        try:
            d = self.delta.predict(x.reshape(1,-1)) -self.e
            return d
        except:
            d = self.delta[self.to_s(x)]
            return d

    def test_q(self,q=None, games=1000): #Tests the current Q function on a number of consecutive games.
        rewards = []
        if q is None:
            q = self.q
        trns = 0
        lns  = []
        rew = 0
        for i in range(games):
            done = False
            self.reset()
            lengame = 0
            while not done:
                state = self.env.global_state()
                action = self.epsilon_greedy_q_policy(self.eval_q(q,state),0.01)
                statenew, reward, done, info = self.env.step(action)
                rew *= self.gamma
                rew += reward
                trns += 2
                lengame += 1
                rewards.append(reward*(self.gamma**len(rewards)))
            lns.append(lengame)
        return sum(rewards), trns/games, sum(lns)/len(lns)

    def test_q_delta(self, games=1000): # Tests the current Q function with event based communication.
        rewards = []
        lns = []
        self.transmissions = 0
        preytriggers = [1 for a in self.env.preys]
        for j in range(games):
            self.reset()
            done = False
            ln = 0
            delayed_state = self.env.global_state()
            N = len(self.env.agents)
            # Initialise agent delayed states
            for agent in self.env.agents:
                agent.set_delayed_state(np.concatenate([agent.state] + [a.state for a in self.env.preys]))
            while not done:
                # Check trigger condition
                trigger = [0 for i in range(N)]
                gamma_value = self.gamma_eval(delayed_state)
                preystates = [a.state for a in self.env.preys]
                for ti,agent in enumerate(self.env.agents):
                    agent_measurement = np.concatenate([agent.state]+preystates)
                    if np.linalg.norm(np.add(agent_measurement,-agent.delayed_state),np.inf)>=gamma_value:
                        # Trigger communication and update agent state
                        trigger[ti] = 1
                        agent.set_delayed_state(np.concatenate([agent.state] + preystates))
                # Update delayed state
                trigger2 = trigger + preytriggers
                delayed_state = np.concatenate([a.state if trigger2[ia] else a.delayed_state[0:2] for ia,a in enumerate(self.env.agents+self.env.preys)])
                # Do greedy action using the delayed state
                action = self.epsilon_greedy_q_policy(self.eval_q(self.q,delayed_state),0.01)
                statenew, reward, done, info = self.env.step(action)
                ln += 1
                rewards.append(reward*(self.gamma**len(rewards)))
                self.transmissions += sum(trigger)
            lns.append(ln)
        return sum(rewards), self.transmissions/games, sum(lns) / len(lns)

    def _doubleq_update_experience_replay(self,q='A'):
        # Perform the doubleQ-learning update rule using the experience replay buffer B.
        nexp = random.sample(self.B, min(len(self.B),50))
        if q=='A':
            for item in nexp:
                step = self.alpha * (
                        item[2] + self.gamma * (not item[4]) * self.eval_q(self.qB,item[3]).max() - self.eval_q(self.qA,np.concatenate([item[0], item[1]])))
                self.qA[tuple(np.concatenate([item[0], item[1]]))] += step
        else:
            for item in nexp:
                step = self.alpha * (
                        item[2] + self.gamma * (not item[4]) * self.eval_q(self.qA,item[3]).max() - self.eval_q(self.qB,np.concatenate([item[0], item[1]])))
                self.qB[tuple(np.concatenate([item[0], item[1]]))] += step

    def train_doubleQ(self,epochs=500000,eps_decay=0.99999):
        cumulative_rew = []
        if self.q is None:
            self.qA = np.dot(np.random.random_sample(self.env.state_action_space),0.1)
            self.qB = np.dot(np.random.random_sample(self.env.state_action_space),0.1)
        else:
            self.qA = self.q
            self.qB = np.copy(self.q)
        for i in range(epochs):
            done = False
            self.env.reset()
            while not done:
                state = self.env.global_state()
                action = self.epsilon_greedy_q_policy(self.eval_q(self.qA,state),epsilon=self.epsilon)
                statenew, reward, done, info = self.env.step(action)
                self.B.append((state,action,reward,statenew,done))
            #Train
            self._doubleq_update_experience_replay('A')
            self._doubleq_update_experience_replay('B')
            if i%1000 == 0:
                rw,_,_ = self.test_q(self.qA,games=1)
                cumulative_rew.append(rw)
                self.cr *= 0.9
                self.cr += 0.1*rw
                print('Epoch = '+str(i)+', Reward = '+str(self.cr))
        self.q = np.copy(self.qA)
        self.qA = None
        self.qB = None
        return cumulative_rew

    def _valid_action_policy(self,sens=0.01):
        q = []
        for i in range(self.env.nS):
            qvals = self.q[i]
            maxq = np.max(qvals)
            q += np.where(qvals >= maxq-sens)
        return q

    def _generate_neighbours(self,s,d):
        bounds = [d,-1*d]
        states = []
        base = 2*d+1
        numofstates = (2*d+1)**len(s)
        s0 = np.copy(s)
        for i in range(len(s0)):
            s0[i] = s0[i]+bounds[1]
        for i in range(numofstates):
            newstate = self.s_to_vec(i,base=base)
            states.append(newstate)
        self.neighbours[d] = states
        return states

    def compute_gamma_point(self, sensitivity, xi):
        qvals = self.eval_q(self.q,xi)
        optimalacts = self.optimal_acts(qvals)
        d = 0
        boxsize = np.asarray(self.env.state_shape)-1
        cond = False
        while not cond:
            d += 1
            # generate states d-far
            try:
                S = self.neighbours[d]
            except:
                print("Computing d=" + str(d))
                S = self._generate_neighbours(xi, d)
            for si in S:
                snew = np.add(si, xi)
                snew[snew<0]=0
                snew[np.add(boxsize,-snew)<0]=boxsize[np.add(boxsize,-snew)<0]
                qx = self.eval_q(self.q, snew)
                maxqx = np.max(qx)
                if any([qx[a] < maxqx - sensitivity for a in optimalacts]) or d > 4:
                    cond = True
                    break
        print(str(xi) + " State. Delta=" + str(d))
        return d

    def compute_gamma_exact(self, sensitivity=0.01):
        delta = np.ones(self.env.state_shape)
        for i in range(self.env.nS):
            s = self.s_to_vec_base(i,base=self.env.state_shape)
            delta[tuple(s)] = self.compute_gamma_point(sensitivity, s)
        return delta

    @staticmethod
    def s_to_vec(s,base=10):
        return np.asarray([int(s/base**5)%(base**1),int(s/base**4)%(base**1),int(s/base**3)%(base**1),int(s/base**2)%(base**1),int(s/base**1)%(base**1),s%(base**1)])

    @staticmethod
    def s_to_vec_base(s,base=[5, 5]):
        return np.asarray([int(s / base[-1-i] ** (len(base)-1-i)) % (base[-1-i] ** 1) for i in range(len(base))])

    @staticmethod
    def optimal_acts(q):
        acts = []
        a1 = np.nonzero(q == q.max())
        for i in range(len(a1[0])):
            acts.append(tuple([a1[k][i] for k in range(len(a1))]))
        return acts

    @staticmethod
    def eval_q(q,state):
        return q[tuple(state)]

    @staticmethod
    def epsilon_greedy_q_policy(qvalues,epsilon=1):
        rndmzr = random.uniform(0, 1)
        # Epsilon-Greedy action
        if rndmzr < epsilon:
            a = np.asarray([np.random.choice(qvalues.shape[i]) for i in range(len(qvalues.shape))])
        else:
            a1 = np.nonzero(qvalues == qvalues.max())
            rnd = np.random.choice(len(a1[0]))
            a = np.asarray((a1[0][rnd], a1[1][rnd]))
        return a

    @staticmethod
    def greedy_q_policy(qvalues):
        a1 = np.nonzero(qvalues == qvalues.max())
        a = np.asarray((a1[0][0], a1[1][0]))
        return a

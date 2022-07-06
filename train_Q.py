import argparse
from particle_tag.Simulator import MultiAgentParticle
import pickle
import os

EPISODES = 1000000
DISCOUNT_RATE = 0.97
LEARNING_RATE = 0.005
Nagents = 2
mapname = "5x5"


def main(preQ = None, episodes = EPISODES):
    MARL = MultiAgentParticle(DISCOUNT_RATE, LEARNING_RATE, Nagents, map_name=mapname, reward=0)
    try:
        with open(preQ, 'rb') as f:
            data = pickle.load(f)
            MARL.load_Q(data['q'])
    except:
        print("No pre-trained Q")

    MARL.reset_epsilon()
    print(MARL.env.state_shape)
    rewards = MARL.train_doubleQ(epochs=episodes,eps_decay=1)

    if not os.path.exists(os.path.join('.','Qs')):
        os.makedirs(os.path.join('.','Qs'))
    with open(os.path.join('.','Qs','trained_Q_'+mapname+'.pickle'),'wb') as f:
        results = {'q':MARL.q,'rewards':rewards,
                   'parameters':{'g':DISCOUNT_RATE, 'a':LEARNING_RATE, 'N':Nagents, 'EPS':episodes, 'map':mapname}}
        pickle.dump(results,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains centralised Q function on Particle Tag game.')
    parser.add_argument('--preQ', metavar='pre_trained_q', type=str, default='None',
                        help='File with pre-trained Q function.')
    parser.add_argument('--episodes', metavar='episodes', type=int, default=100000,
                        help='Number of learning episodes (default is 100000).')
    args = parser.parse_args()
    main(preQ = args.preQ, episodes = args.episodes)

from particle_tag.Simulator import MultiAgentParticle
import pickle
import argparse
import sys
import os

import numpy as np
from sklearn.svm import NuSVR,SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main(gam, alpha, nu, C, q):
    # Initialize agent with parameters
    with open(q, 'rb') as f:
        data = pickle.load(f)
    MARL = MultiAgentParticle(data['parameters']['g'], data['parameters']['a'], data['parameters']['N'],
                              map_name=data['parameters']['map'])
    MARL.load_Q(data['q'])
    # Generate population of X
    Nsamples = 10000
    X = np.zeros((Nsamples,len(MARL.s_to_vec(1))),dtype=int)
    Y = np.zeros(Nsamples)
    states = np.random.randint(0,MARL.env.nS,Nsamples)
    for i,s in enumerate(states):
        X[i] = MARL.s_to_vec_base(s,base=MARL.env.state_shape)
        Y[i] = MARL.compute_gamma_point(sensitivity=alpha, xi=X[i])
    regr = make_pipeline(StandardScaler(), NuSVR(C=C, nu=nu,kernel='rbf',gamma=gam,verbose=11))
    res = regr.fit(X, Y)
    with open('temp.pickle', 'wb+') as f:
        results = {'q': MARL.q, 'delta': res, 'sens': alpha, 's': s,'X':X,'Y':Y,'gam':gam,
                   'parameters': {'g': data['parameters']['g'], 'a': data['parameters']['a'],
                                  'N': data['parameters']['N'], 'EPS': data['parameters']['EPS'],'map': data['parameters']['map']}}
        pickle.dump(results, f)



if __name__ == '__main__':
    with open('temp.pickle','rb') as f:
        params = pickle.load(f)
    os.remove('temp.pickle')
    main(params['parameters']['g'], params['parameters']['a'], params['parameters']['nu'], params['parameters']['c'],
         params['parameters']['q'])

from particle_tag.Simulator import MultiAgentParticle
import os
import pickle
import argparse


def main(alpha, q):
    # Initialize agent with parameters
    with open(q, 'rb') as f:
        data = pickle.load(f)
    MARL = MultiAgentParticle(data['parameters']['g'], data['parameters']['a'], data['parameters']['N'], map_name=data['parameters']['map'])
    MARL.load_Q(data['q'])
    MARL.reset_epsilon()
    delta = MARL.compute_gamma_exact(sensitivity=alpha)
    if not os.path.exists(os.path.join('.','robustness_surrogates')):
        os.makedirs(os.path.join('.','robustness_surrogates'))
    with open(os.path.join('.','robustness_surrogates','rs_exact_'+str(alpha)+'.pickle'),'wb+') as f:
        results = {'q':MARL.q,'delta':delta, 'sens':alpha,
                   'parameters':{'g':data['parameters']['g'],'a':data['parameters']['a'],
                                 'N':data['parameters']['N'], 'EPS':data['parameters']['EPS']},
                                 'map':data['parameters']['map']}
        pickle.dump(results,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes exact value of Gamma function.')
    parser.add_argument('--preQ', metavar='pre_trained_q', type=str, default=os.path.join('Qs','trained_Q_10x10.pickle'),
                        help='File with pre-trained Q function')
    parser.add_argument('--alpha', metavar='sensitivity_parameter', type=float, default=0.1,
                        help='Sensitivity (alpha) in robustness surrogate')
    args = parser.parse_args()
    main(args.alpha,args.preQ)



from particle_tag.Simulator import MultiAgentParticle
import pickle
import argparse
import sys
import subprocess
import os


def sstar(model,e,X,Y):
    ypred = model.predict(X)
    s=0
    for yi,ypredi in zip(Y,ypred):
        if abs(yi-ypredi)>float(e):
            s+=1
    return s


def readkappa():
    with open("output_svr.txt",'r') as f:
        text = f.readlines()
    l = 0
    eps=0
    while l<len(text):
        line = text[-1-l]
        if "epsilon" in line:
            pos = line.find("= ")
            eps = line[pos+2:-1]
            break
        l += 1
    return round(float(eps)*10000)/10000  # We round the resulting epsilon to account for the model sensitivity.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Computes Gamma function from SVR Approximation.')
    parser.add_argument('--preQ', metavar='pre_trained_q', type=str, default=os.path.join('Qs','trained_Q_10x10.pickle'),
                        help='File with pre-trained Q function')
    parser.add_argument('--alpha', metavar='sensitivity_parameter', type=float, default=0.1,
                        help='Sensitivity (alpha) in robustness surrogate')
    parser.add_argument('--nu', metavar='nu_svr', type=float, default=0.1,
                        help='Parameter nu for nu-SVR.')
    parser.add_argument('--C', metavar='C_svr', type=float, default=100,
                        help='Parameter C for SVR.')
    parser.add_argument('--gamma', metavar='gamma_svr', type=float, default=1,
                        help='Parameter gamma for SVR.')
    args = parser.parse_args()
    # Due to a bug in scikit-learn, we can only read the epsilon value from the verbose prints (it is not stored
    # internally in the model). Therefore, we need to read and write files locally several times, to allow subroutines
    # to capture the verbose prints and save the epsilon value.
    with open('temp.pickle', 'wb+') as f:
        params = {'parameters': {'g': args.gamma, 'a': args.alpha,
                                  'nu': args.nu, 'c': args.C,'q':args.preQ}}
        pickle.dump(params, f)
    with open("output_svr.txt", "wb") as f:
        subprocess.check_call(["python", "_robust_svr.py"], stdout=f)
    with open('temp.pickle','rb') as f:
        data = pickle.load(f)
    os.remove('temp.pickle')
    # As mentioned, here we read the kappa value from the stored text output (called epsilon in the SVR module).
    k = readkappa()
    print("kappa="+str(k))
    # We compute the s* value (see page 9 in the paper)
    s = sstar(data['delta'], k, data['X'], data['Y'])
    print(data['parameters'])
    if not os.path.exists(os.path.join('.','robustness_surrogates')):
        os.makedirs(os.path.join('.','robustness_surrogates'))
    with open(os.path.join('.','robustness_surrogates','rs_SVR_nu' + str(args.nu) + '.pickle'), 'wb+') as f:
        results = {'q': data['q'], 'delta': data['delta'], 'sens': args.alpha, 'e': k, 's': s,
                   'parameters': {'g': data['parameters']['g'], 'a': data['parameters']['a'],
                                  'N': data['parameters']['N'], 'EPS': data['parameters']['EPS'],
                                  'map': data['parameters']['map']}}
        pickle.dump(results, f)


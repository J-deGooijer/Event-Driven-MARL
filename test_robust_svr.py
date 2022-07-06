from particle_tag.Simulator import MultiAgentParticle
import pickle
import numpy as np
import os
import argparse

SAMPLE_SIZE = 500

def test_gamma(fnam):
    # Initialize agent with parameters
    with open(fnam, 'rb') as f:
        data = pickle.load(f)
    MARL = MultiAgentParticle(data['parameters']['g'], data['parameters']['a'], data['parameters']['N'], map_name=data['parameters']['map'], reward=0)
    MARL.load_Q(data['q'])
    MARL.load_delta(data['delta'])
    try:
        MARL.load_e(data['e'])
    except:
        print("No epsilon.")
    results =[]
    avglenet = []
    trnset = []
    for i in range(SAMPLE_SIZE):
        resultsi,trnseti, avgleneti = MARL.test_q_delta(games=100)
        results.append(resultsi)
        avglenet.append(avgleneti)
        trnset.append(trnseti)
    meanet = sum(results)/SAMPLE_SIZE
    varet = np.var(results)
    meanavglenet = sum(avglenet)/SAMPLE_SIZE
    varavglenet = np.var(avglenet)
    meantrnset = sum(trnset)/SAMPLE_SIZE
    vartrnset = np.var([i for i in trnset])
    print('Results for SVR with alpha='+str(data['sens']))
    print("g: "+str(meanavglenet))
    print("Var(g): "+str(varavglenet))
    print("Rewards: "+str(meanet))
    print("Var(Rewards): "+str(varet))
    print("h(T): "+str((meantrnset) ))
    print("Var(h(T)): "+str(vartrnset))
    return results,avglenet,trnset

def test_continuous(fnam):
    # Initialize agent with parameters
    with open(fnam, 'rb') as f:
        data = pickle.load(f)
    MARL = MultiAgentParticle(data['parameters']['g'], data['parameters']['a'], data['parameters']['N'],
                              map_name=data['parameters']['map'], reward=0)
    MARL.load_Q(data['q'])
    results_old = []
    avglen = []
    trnscont = []
    for i in range(SAMPLE_SIZE):
        results_oldi, trnsconti,avgleni = MARL.test_q(games=100)
        results_old.append(results_oldi)
        avglen.append(avgleni)
        trnscont.append(trnsconti)
    meanper = sum(results_old)/SAMPLE_SIZE
    varper = np.var(results_old)
    meanavglen = sum(avglen)/SAMPLE_SIZE
    varavglen = np.var(avglen)
    meantrnscont = sum(trnscont)/SAMPLE_SIZE
    vartran = np.var(trnscont)
    print("g (continuous): " + str(meanavglen))
    print("Var(g) (continuous): " + str(varavglen))
    print("Rewards (continuous): " + str(meanper))
    print("Var(Rewards) (continuous): " + str(varper))
    print("h(T) (continuous): " + str(meantrnscont))
    print("Var(h(T)) (continuous): "+ str(vartran))
    return results_old,avglen,trnscont

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs tests for different robustness surrogate functions computed as SVRs.')
    parser.add_argument('--directoryname', metavar='dirname', type=str,
                        help='directory where the SVRs are loaded from')
    args = parser.parse_args()
    results = {}
    try:
        dirc = args.directoryname
        files = os.listdir(args.directoryname)
        path = os.path.join(dirc,files[0])
    except Exception as e:
        print(e)
        dirc = 'robustness_surrogates'
        files = os.listdir('robustness_surrogates')
        path = os.path.join(dirc,files[0])
    # Compute continuous communication
    for f in files:
        path = os.path.join(dirc,f)
        res = {'et':{},'cont':{}}
        print("---------------------------------------------")
        print("Computing continuous communication baseline for file:")
        print(path)
        res['cont'] = test_continuous(path)
        print("---------------------------------------------")
        print("Computing event driven communication results for file:")
        print(path)
        res['et']=test_gamma(path)
        results[f] = res

    with open('aux_files/results_test_gamma.pickle', 'wb+') as f:
        pickle.dump(results,f)


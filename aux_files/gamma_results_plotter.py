import pickle
import numpy as np
import matplotlib.pyplot as plt

import tikzplotlib

color0 = (0.12156862745098,0.466666666666667,0.705882352941177)
color1 = (1,0.498039215686275,0.0549019607843137)
color2 = (0.172549019607843,0.627450980392157,0.172549019607843)
color3 = (0.83921568627451,0.152941176470588,0.156862745098039)
color4 = (0.580392156862745,0.403921568627451,0.741176470588235)
color5 = (0.549019607843137,0.337254901960784,0.294117647058824)
with open('results_test_gamma.pickle', 'rb') as f:
    data = pickle.load(f)

data.pop('delta_SVR_nu0.1_alpha0.5.pickle',None)
alphas = [0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
rews = []
comms = []
leng = []
rews.append(data[0][0])
comms.append(data[0][2])
leng.append(data[0][1])

data.pop(0,None)
for a in alphas:
    for k in data.keys():
        if 'alpha'+str(a)+'.pickle' in k:
            rews.append(data[k][0])
            comms.append(data[k][2])
            leng.append(data[k][1])
            break
alphas = [0,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
fig1, ax1 = plt.subplots()
ax1.set_title('Rewards - ET Communication')
ax1.boxplot(rews,positions=alphas,widths=0.08)
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel('Rewards')
tikzplotlib.save("rews_boxplot.tex")

fig1, ax1 = plt.subplots()
ax1.set_title('Messages - ET Communication')
ax1.boxplot(comms,positions=alphas,widths=0.08)
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel('Messages per Game')
tikzplotlib.save("comms_boxplot.tex")
for a,r,c,l in zip(alphas,rews,comms,leng):
    print("alpha="+str(a))
    print("rewards avg=" + str(np.mean(r)))
    print("rewards std=" + str(np.sqrt(np.var(r))))
    print("comms avg=" + str(np.mean(c)))
    print("comms std=" + str(np.sqrt(np.var(c))))
    print("length games avg=" + str(np.mean(l)))
    print("length games std=" + str(np.sqrt(np.var(l))))
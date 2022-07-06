import pickle
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm

import tikzplotlib

color0 = (0.12156862745098,0.466666666666667,0.705882352941177)
color1 = (1,0.498039215686275,0.0549019607843137)
color2 = (0.172549019607843,0.627450980392157,0.172549019607843)
color3 = (0.83921568627451,0.152941176470588,0.156862745098039)
color4 = (0.580392156862745,0.403921568627451,0.741176470588235)
color5 = (0.549019607843137,0.337254901960784,0.294117647058824)
with open('data_embeddings.pickle', 'rb') as f:
    data = pickle.load(f)

grid_x, grid_y = np.mgrid[-40:40:100j, -40:40:100j]

grid_z05 = griddata(data['X_emb'], data['Y'][0.5], (grid_x, grid_y), method='cubic')
grid_z06 = griddata(data['X_emb'], data['Y'][0.6], (grid_x, grid_y), method='cubic')
grid_z07 = griddata(data['X_emb'], data['Y'][0.7], (grid_x, grid_y), method='cubic')
grid_z08 = griddata(data['X_emb'], data['Y'][0.8], (grid_x, grid_y), method='cubic')


fig = plt.figure(figsize=plt.figaspect(0.8))
ax4 = fig.add_subplot(2, 2, 1)
ax4.set_title(r'$\alpha = 0.5$')
im = ax4.imshow(grid_z05.T, extent=(-40,40,-40,40), origin='lower',cmap=cm.inferno,vmin=-0.1,vmax=9)
ax4.axis('off')

ax4 = fig.add_subplot(2, 2, 2)
ax4.set_title(r'$\alpha = 0.6$')
im = ax4.imshow(grid_z06.T, extent=(-40,40,-40,40), origin='lower',cmap=cm.inferno,vmin=-0.1,vmax=9)
ax4.axis('off')

ax4 = fig.add_subplot(2, 2, 3)
ax4.set_title(r'$\alpha = 0.7$')
im = ax4.imshow(grid_z07.T, extent=(-40,40,-40,40), origin='lower',cmap=cm.inferno,vmin=-0.1,vmax=9)
ax4.axis('off')

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title(r'$\alpha = 0.8$')
im = ax4.imshow(grid_z08.T, extent=(-40,40,-40,40), origin='lower',cmap=cm.inferno,vmin=-0.1,vmax=9)
ax4.axis('off')

fig.tight_layout()
cbar_ax = fig.add_axes([0.85, 0.2, 0.025, 0.6])
fig.colorbar(im,cax=cbar_ax)
fig.subplots_adjust(right=0.8)
tikzplotlib.save("gamma_embedding.tex")
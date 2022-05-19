import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tikzplotlib import save as tikz_save

load_file_str = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'logs', 'prediction_1_results', '3stepcontinueMDP.pkl')
print(load_file_str)
with open(load_file_str, 'rb') as f:
    load_dict = pickle.load(f)
    mdp = load_dict['mdp']
    x_sa = load_dict['x_sa']
    obstacles = load_dict['obstacles']
    end_states = load_dict['end_states']

num_of_rows = mdp.Nrows
num_of_cols = mdp.Ncols

x_s = np.zeros(mdp.NS)
x_sa_ind = 0
for state in range(mdp.NS):
    num_of_acts = mdp.NA_list[0, state]
    x_s[state] = sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)])
    x_sa_ind = x_sa_ind + num_of_acts

obstacle_matrix = np.zeros((num_of_rows, num_of_cols))
for obstacle in obstacles:
    obstacle_matrix[obstacle[0], obstacle[1]] = 1

target_matrix = np.zeros((num_of_rows, num_of_cols))
for state in end_states:
    loc = np.asarray(np.unravel_index(state, (num_of_rows, num_of_cols)))
    target_matrix[loc[0],loc[1]] = 1


x_s_ind = 0

occ_vars = x_s[x_s_ind:(x_s_ind + mdp.NS)]
occ_vars = np.reshape(occ_vars, (num_of_rows, num_of_cols))
occ_vars[obstacle_matrix == 1] = -np.inf
occ_vars[target_matrix == 1] = -10
occ_vars = np.flip(occ_vars, 0)

print(occ_vars)
current_cmap = plt.cm.Reds
current_cmap.set_bad(color='orange')
shw = plt.imshow(occ_vars, cmap=current_cmap, interpolation='nearest', vmin = -0.05, vmax = 1)
shw.cmap.set_under('green')
plt.axis('off')
#tikz_save('colormap.tex', strict=True)
plt.colorbar(shw)
plt.show()

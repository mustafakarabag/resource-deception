import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

load_file_str = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'logs', 'time_product_mdp_results', '3stepcontinueMDP.pkl')
with open(load_file_str, 'rb') as f:
    load_dict = pickle.load(f)
    time_product_mdp = load_dict['time_product_mdp']
    x_sa = load_dict['x_sa']

T_vis = time_product_mdp.T + 1
num_of_rows = time_product_mdp.original_mdp.Nrows
num_of_cols = time_product_mdp.original_mdp.Ncols

x_s = np.zeros(time_product_mdp.NS)
x_sa_ind = 0
for state in range(time_product_mdp.NS):
    num_of_acts = time_product_mdp.NA_list[0, state]
    x_s[state] = sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)])
    x_sa_ind = x_sa_ind + num_of_acts

x_s_ind = 0
for t in range(T_vis):
    occ_vars = x_s[x_s_ind:(x_s_ind + time_product_mdp.original_mdp.NS)]
    occ_vars = np.reshape(occ_vars, (num_of_rows, num_of_cols))
    plt.imshow(occ_vars, cmap='hot', interpolation='nearest')
    plt.show()
    x_s_ind = x_s_ind + time_product_mdp.original_mdp.NS
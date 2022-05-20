import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import markov_decision_processes

load_file_str = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'logs', 'deception_results', '3stepcontinueMDP.pkl')
print(load_file_str)
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
    if state == 100:
        print('4')
    num_of_acts = time_product_mdp.NA_list[0, state]
    x_s[state] = sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)])
    x_sa_ind = x_sa_ind + num_of_acts

x_s_ind = 0
for t in range(T_vis):
    occ_vars = x_s[x_s_ind:(x_s_ind + time_product_mdp.original_mdp.NS)]
    occ_vars = np.reshape(occ_vars, (num_of_rows, num_of_cols))
    print(occ_vars)
    shw = plt.imshow(occ_vars, cmap=plt.cm.Reds, interpolation='nearest')
    plt.colorbar(shw)
    plt.show()
    x_s_ind = x_s_ind + time_product_mdp.original_mdp.NS

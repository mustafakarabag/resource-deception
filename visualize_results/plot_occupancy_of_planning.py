import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

load_file_str = os.path.join(os.path.dirname(os.path.abspath(os.path.curdir)), 'logs', 'deception_results', '3stepcontinueMDP.pkl')
print(load_file_str)
with open(load_file_str, 'rb') as f:
    load_dict = pickle.load(f)
    time_product_mdp = load_dict['time_product_mdp']
    x_sa = load_dict['x_sa']
    obstacles = load_dict['obstacles']
    end_states = load_dict['end_states']

T_vis = time_product_mdp.T + 1
num_of_rows = time_product_mdp.original_mdp.Nrows
num_of_cols = time_product_mdp.original_mdp.Ncols

x_s = np.zeros(time_product_mdp.NS)
x_sa_ind = 0
for state in range(time_product_mdp.NS):
    num_of_acts = time_product_mdp.NA_list[0, state]
    x_s[state] = sum(x_sa[0,x_sa_ind:(x_sa_ind+num_of_acts)])
    x_sa_ind = x_sa_ind + num_of_acts

obstacle_matrix = np.zeros((num_of_rows, num_of_cols))
for obstacle in obstacles:
    obstacle_matrix[obstacle[0], obstacle[1]] = 1

target_matrix = np.zeros((num_of_rows, num_of_cols))
for state in end_states:
    loc = np.asarray(np.unravel_index(state, (num_of_rows, num_of_cols)))
    target_matrix[loc[0],loc[1]] = 1


print(obstacle_matrix)
x_s_ind = 0
total_occupancy = np.zeros((num_of_rows, num_of_cols))
for t in range(T_vis):
    occ_vars = np.round(x_s[x_s_ind:(x_s_ind + time_product_mdp.original_mdp.NS)],3)
    occ_vars = np.reshape(occ_vars, (num_of_rows, num_of_cols))
    occ_vars[obstacle_matrix == 1] = -100
    occ_vars[target_matrix == 1] = -10
    occ_vars = np.flip(occ_vars, 0)

    current_cmap = plt.cm.Reds
    current_cmap.set_bad(color='orange')
    shw = plt.imshow(occ_vars, cmap=current_cmap, interpolation='nearest', vmin = -0.05)
    shw.cmap.set_under('green')

    #plt.imshow(obstacle_matrix)
    #plt.colorbar(shw)
    #plt.show()
    x_s_ind = x_s_ind + time_product_mdp.original_mdp.NS
    total_occupancy += occ_vars

df1 = pd.DataFrame(total_occupancy)
df1.reset_index(drop=True, inplace=True)
df1.to_csv('sample.csv', index = False, header = None)
current_cmap = plt.cm.Reds
current_cmap.set_bad(color='orange')
shw = plt.imshow(total_occupancy, cmap=current_cmap, interpolation='nearest', vmin = -0.05)
shw.cmap.set_under('green')
plt.colorbar(shw)
plt.show()

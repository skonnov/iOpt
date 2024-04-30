from os import listdir
from os.path import isfile, join
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

path_to_files = 'D:/Works/aspirantura/cluster_data/30_04_24'

files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]


functions = set()


data = {}

for file in files:
    with open(path_to_files + '/' + file) as f:
        input = f.readline() # skip first line

        cur_e = None
        cur_func = None
        cur_alpha = None

        input = f.readline().split()
        while input:
            if input[0] == 'f':
                functions.add((int(input[1]), int(input[2])))
            if input[0] == 'e':
                cur_e = input[1]
            if input[0].startswith('d_'):
                cur_func = input[0]
            if input[0] == 'hw':
                hw = float(input[1])
                num_of_iter = float(input[3])
                if cur_func == 'd_prob' or cur_func == 'd_prob_poly' or cur_func == 'd_prob_rbf':
                    if cur_func not in data:
                        data[cur_func] = {}
                    if cur_alpha not in data[cur_func]:
                        data[cur_func][cur_alpha] = {}
                    if cur_e not in data[cur_func][cur_alpha]:
                        data[cur_func][cur_alpha][cur_e] = {'hw': [], 'num_of_iter': []}

                    data[cur_func][cur_alpha][cur_e]['hw'].append(hw)
                    data[cur_func][cur_alpha][cur_e]['num_of_iter'].append(num_of_iter)
                else:
                    if cur_func not in data:
                        data[cur_func] = {}
                    if cur_e not in data[cur_func]:
                        data[cur_func][cur_e] = {'hw': [], 'num_of_iter': []}
                    data[cur_func][cur_e]['hw'].append(hw)
                    data[cur_func][cur_e]['num_of_iter'].append(num_of_iter)
            if input[0] == 'a':
                cur_alpha = input[1]
            input = f.readline().split()


print(len(functions))

average_data = {}
for func in data:
    average_data[func] = {}
    if func == 'd_prob' or func == 'd_prob_poly' or func == 'd_prob_rbf':
        # average_data[func]['0.01'] = {}
        # average_data[func]['0.02'] = {}
        for alpha in data[func]:
            average_data[func][alpha] = {}
            for e in data[func][alpha]:
                average_data[func][alpha][e] = {}
                average_data[func][alpha][e]['hw'] = mean(data[func][alpha][e]['hw'])
                average_data[func][alpha][e]['num_of_iter'] = mean(data[func][alpha][e]['num_of_iter'])
    else:
        average_data[func] = {}
        for e in data[func]:
            average_data[func][e] = {}
            average_data[func][e]['hw'] = mean(data[func][e]['hw'])
            average_data[func][e]['num_of_iter'] = mean(data[func][e]['num_of_iter'])

print(average_data)


mgsa_hw = [average_data['d_mgsa'][e]['hw'] for e in average_data['d_mgsa']]
mgsa_num_of_iter = [average_data['d_mgsa'][e]['num_of_iter'] for e in average_data['d_mgsa']]

mgsa_dist_hw = [average_data['d_appr'][e]['hw'] for e in average_data['d_appr']]
mgsa_dist_num_of_iter = [average_data['d_appr'][e]['num_of_iter'] for e in average_data['d_appr']]

mgsa_prob_hw = {}
mgsa_prob_num_of_iter = {}

mgsa_prob_poly_hw = {}
mgsa_prob_poly_num_of_iter = {}

mgsa_prob_rbf_hw = {}
mgsa_prob_rbf_num_of_iter = {}

if 'd_prob' in average_data:
    for alpha in average_data['d_prob']:
        # continue
        # if alpha != "0.07" and alpha != "0.08":
        # if alpha != "0.07":
        # if alpha == "0.01" or alpha == "0.02" or alpha == "0.15" or alpha == "0.2":
        if alpha != "0.03" and alpha != "0.09":
            continue
        mgsa_prob_hw[alpha] = [average_data['d_prob'][alpha][e]['hw'] for e in average_data['d_prob'][alpha]]
        mgsa_prob_num_of_iter[alpha] = [average_data['d_prob'][alpha][e]['num_of_iter'] for e in average_data['d_prob'][alpha]]

if 'd_prob_poly' in average_data:
    for alpha in average_data['d_prob_poly']:
        # continue
        # if alpha != "0.06" and alpha != "0.08":
        # if alpha != "0.05" and alpha != "0.06" and alpha != "0.07" and alpha != "0.08" and alpha != "0.09":
        if alpha != "0.03" and alpha != "0.09":
            continue
        mgsa_prob_poly_hw[alpha] = [average_data['d_prob_poly'][alpha][e]['hw'] for e in average_data['d_prob_poly'][alpha]]
        mgsa_prob_poly_num_of_iter[alpha] = [average_data['d_prob_poly'][alpha][e]['num_of_iter'] for e in average_data['d_prob_poly'][alpha]]

if 'd_prob_rbf' in average_data:
    for alpha in average_data['d_prob_rbf']:
        # continue
        # if alpha != "0.05" and alpha != "0.07":
        # if alpha != "0.05" and alpha != "0.06" and alpha != "0.07":
        if alpha != "0.03" and alpha != "0.09":
            continue
        mgsa_prob_rbf_hw[alpha] = [average_data['d_prob_rbf'][alpha][e]['hw'] for e in average_data['d_prob_rbf'][alpha]]
        mgsa_prob_rbf_num_of_iter[alpha] = [average_data['d_prob_rbf'][alpha][e]['num_of_iter'] for e in average_data['d_prob_rbf'][alpha]]


plt.plot(mgsa_num_of_iter, mgsa_hw, label='MGSA')

plt.plot(mgsa_dist_num_of_iter, mgsa_dist_hw, label = 'MGSA Dist')

for alpha in mgsa_prob_hw:
    plt.plot(mgsa_prob_num_of_iter[alpha], mgsa_prob_hw[alpha], label = 'MGSA Prob Linear, a=' + str(alpha))

for alpha in mgsa_prob_poly_hw:
    plt.plot(mgsa_prob_poly_num_of_iter[alpha], mgsa_prob_poly_hw[alpha], label = 'MGSA Prob Poly, a=' + str(alpha))

for alpha in mgsa_prob_rbf_hw:
    plt.plot(mgsa_prob_rbf_num_of_iter[alpha], mgsa_prob_rbf_hw[alpha], label = 'MGSA Prob Rbf, a=' + str(alpha))

plt.xlabel('Average number of iterations')
plt.ylabel('Average value of HW index')

plt.legend()
plt.show()

fig, ax = plt.subplots(5, 1, sharey='row', sharex='col')

plt.setp(ax.flat, xlabel='alpha', ylabel='avg hw index')
pad = 5 # in points
for axis, e in zip(ax, average_data['d_appr']):
    axis.annotate("e = " + e, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0),
                xycoords=axis.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

# ax[0, 0].set_title("MGSA Prob Linear")
# ax[0, 1].set_title("MGSA Prob Poly")
# ax[0, 2].set_title("MGSA Prob Rbf")

for i, e in enumerate(average_data['d_appr']):
    ax[i].plot([alpha for alpha in mgsa_prob_hw], [mgsa_prob_hw[alpha][i] for alpha in mgsa_prob_hw], label='MGSA Prob Linear')
    ax[i].plot([alpha for alpha in mgsa_prob_poly_hw], [mgsa_prob_poly_hw[alpha][i] for alpha in mgsa_prob_poly_hw], label='MGSA Prob Poly')
    ax[i].plot([alpha for alpha in mgsa_prob_rbf_hw], [mgsa_prob_rbf_hw[alpha][i] for alpha in mgsa_prob_rbf_hw], label='MGSA Prob Rbf')
    # # for axis in ax:
    ax[i].axhline(y=mgsa_hw[i], xmin=0.03, xmax=0.97, color="red", label="MGSA")
    ax[i].axhline(y=mgsa_dist_hw[i], xmin=0.03, xmax=0.97, color="purple", label="MGSA Dist")

ax[0].legend(loc="upper left", bbox_to_anchor=(0, 1.8))

plt.show()


fig, ax = plt.subplots(5, 1, sharey='row', sharex='col')

plt.setp(ax.flat, xlabel='alpha', ylabel='avg N of iter')
pad = 5 # in points
for axis, e in zip(ax, average_data['d_appr']):
    axis.annotate("e = " + e, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0),
                xycoords=axis.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

# ax[0].set_title("MGSA Prob Linear")
# ax[0].set_title("MGSA Prob Poly")
# ax[0].set_title("MGSA Prob Rbf")

for i, e in enumerate(average_data['d_appr']):
    ax[i].plot([alpha for alpha in mgsa_prob_num_of_iter], [mgsa_prob_num_of_iter[alpha][i] for alpha in mgsa_prob_num_of_iter], label='MGSA Prob Linear')
    ax[i].plot([alpha for alpha in mgsa_prob_poly_num_of_iter], [mgsa_prob_poly_num_of_iter[alpha][i] for alpha in mgsa_prob_poly_num_of_iter], label='MGSA Prob Poly')
    ax[i].plot([alpha for alpha in mgsa_prob_rbf_num_of_iter], [mgsa_prob_rbf_num_of_iter[alpha][i] for alpha in mgsa_prob_rbf_num_of_iter], label='MGSA Prob Rbf')
    # for axis in ax:
    ax[i].axhline(y=mgsa_num_of_iter[i], xmin=0.03, xmax=0.97, color="red", label="MGSA")
    ax[i].axhline(y=mgsa_dist_num_of_iter[i], xmin=0.03, xmax=0.97, color="purple", label="MGSA Dist")

ax[0].legend(loc="upper left", bbox_to_anchor=(0, 1.8))

plt.show()

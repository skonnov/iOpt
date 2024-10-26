from os import listdir
from os.path import isfile, join
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

# paths_to_files = ['D:/Works/aspirantura/cluster_data/24_06_16_additional/4_criteria']
paths_to_files = ['D:/Works/aspirantura/cluster_data/25-27_10_17_xgboost']

# redo using bunch of classes instead of difficult list of dict of dict of lists?
# or maybe even use databases? (mongodb???)

def getAverageData(path_to_files):
    files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f))]
    functions = set()
    data = {}

    for file in files:
        with open(path_to_files + '/' + file) as f:
            input = f.readline() # skip first line

            cur_e = None
            cur_model = None
            cur_alpha = None

            input_not_splitted = f.readline()
            input = input_not_splitted.split()
            while input:
                if input[0] == 'a':
                    cur_alpha = input[1]
                if input[0] == 'e':
                    cur_e = input[1]
                if input[0] == 'f':
                    # if (input_not_splitted not in function):
                    #     print("Add new function: ", input_not_splitted)
                    functions.add(input_not_splitted)
                if input[0] == 'm':
                    cur_model = input[1]
                if input[0] == 'hw':
                    hw = float(input[1])
                    num_of_iter = float(input[3])

                    if cur_alpha is None:
                        cur_alpha = -1
                    if cur_e is None:
                        print("e is not presented!")
                        raise ValueError()
                    if cur_model is None:
                        print("function is not presented!")
                        raise ValueError()

                    if cur_model not in data:
                        data[cur_model] = {}
                    if cur_alpha not in data[cur_model]:
                        data[cur_model][cur_alpha] = {}
                    if cur_e not in data[cur_model][cur_alpha]:
                        data[cur_model][cur_alpha][cur_e] = {'hw': [], 'num_of_iter': []}
                    data[cur_model][cur_alpha][cur_e]['hw'].append(hw)
                    data[cur_model][cur_alpha][cur_e]['num_of_iter'].append(num_of_iter)
                input_not_splitted = f.readline()
                input = input_not_splitted.split()

    print("----------> Number of calculated samples: ", len(functions), "<---------")

    average_data = {}
    for func in data:
        average_data[func] = {}
        for alpha in data[func]:
            average_data[func][alpha] = {}
            for e in data[func][alpha]:
                average_data[func][alpha][e] = {}
                average_data[func][alpha][e]['hw'] = mean(data[func][alpha][e]['hw'])
                average_data[func][alpha][e]['num_of_iter'] = mean(data[func][alpha][e]['num_of_iter'])
    print(average_data)
    return average_data

def getAverageDataForPaths(paths_to_files):
    average_datas = []
    for path in paths_to_files:
        average_datas.append(getAverageData(path))
        average_datas[-1]['path'] = path
    return average_datas

def getParsedModelsNames(average_datas):
    models_names = {}
    for average_data in average_datas:
        models_names[average_data['path']] = []
        for model in average_data:
            models_names[average_data['path']].append(model)
    return models_names

def organizeAverageValues(average_datas):
    hw_vals = {}
    num_of_iter_vals = {}
    for average_data in average_datas:
        path = average_data['path']
        print(path)
        for model in average_data:
            if model == 'path':
                continue
            if (model not in hw_vals):
                hw_vals[model] = {}
                num_of_iter_vals[model] = {}
            if path not in hw_vals[model]:
                hw_vals[model][path] = {}
                num_of_iter_vals[model][path] = {}

            average_data[model].items()  # ?
            average_data[model] = dict(sorted(average_data[model].items()))  #?
            for alpha in average_data[model]:
                if alpha not in hw_vals[model][path]:
                    hw_vals[model][path][alpha] = {}
                    num_of_iter_vals[model][path][alpha] = {}
                hw_vals[model][path][alpha] = [average_data[model][alpha][e]['hw'] for e in average_data[model][alpha]]
                num_of_iter_vals[model][path][alpha] = [average_data[model][alpha][e]['num_of_iter'] for e in average_data[model][alpha]]
    return hw_vals, num_of_iter_vals

def show_plot_with_hw_and_num_of_iter(path, hw_vals, num_of_iter_vals):
    plt.grid()

    markers = ['X', 'd', '^', 's']
    for id, model in enumerate(hw_vals):
        if path in hw_vals[model]:
            for alpha in hw_vals[model][path]:
                label = model
                if (float(alpha) > 1e-4):
                    label += ", a = " + alpha
                plt.plot(num_of_iter_vals[model][path][alpha], hw_vals[model][path][alpha], label=label, linewidth=5.0, marker=markers[id % len(markers)], markersize=15, alpha=0.6)
                # plt.plot(mgsa_dist_num_of_iter[path],          mgsa_dist_hw[path],          label = 'ML_MGSA_Dist', linewidth=5.0, alpha=0.6, marker='X', markersize=15)

    plt.xlabel('Average number of iterations', weight='bold')
    plt.ylabel('Average value of HW index', weight='bold')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    average_datas = getAverageDataForPaths(paths_to_files)
    hw_vals, num_of_iter_vals = organizeAverageValues(average_datas)
    for model in hw_vals:
        print("Model: ", model)
        for path in hw_vals[model]:
            print("Path: ", path)
            for alpha in hw_vals[model][path]:
                print("Alpha, ", alpha)
                print("Vals for each e:", hw_vals[model][path][alpha])
                print("Num of iters for each e: ", num_of_iter_vals[model][path][alpha])

    show_plot_with_hw_and_num_of_iter(path, hw_vals, num_of_iter_vals)

# def show_plot_with_hw(path):
#     fig, ax = plt.subplots(5, 1, sharey='row', sharex='col')

#     plt.setp(ax.flat, xlabel='alpha', ylabel='avg hw index')
#     pad = 5 # in points
#     for axis, e in zip(ax, average_data['xgboost']):
#         axis.annotate("e = " + e, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0),
#                     xycoords=axis.yaxis.label, textcoords='offset points',
#                     size='large', ha='right', va='center')

#     ax[0, 0].set_title("MGSA Prob Linear")
#     # ax[0, 1].set_title("MGSA Prob Poly")
#     # ax[0, 2].set_title("MGSA Prob Rbf")

#     for i, e in enumerate(average_data['xgboost']['0.01']):
#         ax[i].plot([alpha for alpha in mgsa_prob_xgboost_hw[path]], [mgsa_prob_xgboost_hw[path][alpha][i] for alpha in mgsa_prob_xgboost_hw[path]], label='MGSA Prob XGBoost')
#         # ax[i].plot([alpha for alpha in mgsa_prob_poly_hw[path]], [mgsa_prob_poly_hw[path][alpha][i] for alpha in mgsa_prob_poly_hw[path]], label='MGSA Prob Poly')
#         # ax[i].plot([alpha for alpha in mgsa_prob_rbf_hw[path]], [mgsa_prob_rbf_hw[path][alpha][i] for alpha in mgsa_prob_rbf_hw[path]], label='MGSA Prob Rbf')
#         # # for axis in ax:
#         # ax[i].axhline(y=mgsa_hw[path][i], xmin=0.03, xmax=0.97, color="red", label="MGSA")
#         # ax[i].axhline(y=mgsa_dist_hw[path][i], xmin=0.03, xmax=0.97, color="purple", label="MGSA Dist")

#     # ax[0].legend(loc="upper left", bbox_to_anchor=(0, 1.8))

# def show_plot_with_num_of_iter(path):
#     fig, ax = plt.subplots(5, 1, sharey='row', sharex='col')

#     plt.setp(ax.flat, xlabel='alpha', ylabel='avg N of iter')
#     pad = 5 # in points
#     for axis, e in zip(ax, average_data['linear_svc']):
#         axis.annotate("e = " + e, xy=(0, 0.5), xytext=(-axis.yaxis.labelpad - pad, 0),
#                     xycoords=axis.yaxis.label, textcoords='offset points',
#                     size='large', ha='right', va='center')

#     # ax[0].set_title("MGSA Prob Linear")
#     # ax[0].set_title("MGSA Prob Poly")
#     # ax[0].set_title("MGSA Prob Rbf")

#     for i, e in enumerate(average_data['linear_svc']):
#         ax[i].plot([alpha for alpha in mgsa_prob_num_of_iter[path]], [mgsa_prob_num_of_iter[path][alpha][i] for alpha in mgsa_prob_num_of_iter[path]], label='MGSA Prob Linear')
#         ax[i].plot([alpha for alpha in mgsa_prob_poly_num_of_iter[path]], [mgsa_prob_poly_num_of_iter[path][alpha][i] for alpha in mgsa_prob_poly_num_of_iter[path]], label='MGSA Prob Poly')
#         ax[i].plot([alpha for alpha in mgsa_prob_rbf_num_of_iter[path]], [mgsa_prob_rbf_num_of_iter[path][alpha][i] for alpha in mgsa_prob_rbf_num_of_iter[path]], label='MGSA Prob Rbf')
#         # for axis in ax:
#         ax[i].axhline(y=mgsa_num_of_iter[path][i], xmin=0.03, xmax=0.97, color="red", label="MGSA")
#         ax[i].axhline(y=mgsa_dist_num_of_iter[path][i], xmin=0.03, xmax=0.97, color="purple", label="MGSA Dist")

#     ax[0].legend(loc="upper left", bbox_to_anchor=(0, 1.8))

# plt.rcParams.update({'font.size': 18})

# for average_data in average_datas:
#     path = average_data['path']

#     show_plot_with_hw_and_num_of_iter(path)

#     # show_plot_with_hw(path)

#     # show_plot_with_num_of_iter(path)

#     plt.show()

# # for path in paths_to_files:
# #     if path == 'D:/Works/aspirantura/cluster_data/24_05_01-24_05_02/adj_weights_log_normalized':
# #         plt.plot(mgsa_num_of_iter[path], mgsa_hw[path], label='MGSA')
# #         plt.plot(mgsa_dist_num_of_iter[path], mgsa_dist_hw[path], label = 'MGSA Dist')
# #         for alpha in ('0.01', '0.04'):
# #             # plt.plot(mgsa_prob_num_of_iter[path][alpha], mgsa_prob_hw[path][alpha], label = 'MGSA Prob Linear, a=' + str(alpha) + ", log_norm")
# #             # plt.plot(mgsa_prob_poly_num_of_iter[path][alpha], mgsa_prob_poly_hw[path][alpha], label = 'MGSA Prob Poly, a=' + str(alpha) + ", log_norm")
# #             plt.plot(mgsa_prob_rbf_num_of_iter[path][alpha], mgsa_prob_rbf_hw[path][alpha], label = 'MGSA Prob Rbf, a=' + str(alpha) + ", log_norm")
# #     # elif path == 'D:/Works/aspirantura/cluster_data/24_05_01-24_05_02/adj_weights_log':
# #     #     for alpha in mgsa_prob_num_of_iter[path]:
# #     #         if alpha == '0.09' or alpha == '0.02' or alpha == '0.03':
# #     #             continue
# #     #         # plt.plot(mgsa_prob_num_of_iter[path][alpha], mgsa_prob_hw[path][alpha], label = 'MGSA Prob Linear, a=' + str(alpha) + ", log")
# #     #         # plt.plot(mgsa_prob_poly_num_of_iter[path][alpha], mgsa_prob_poly_hw[path][alpha], label = 'MGSA Prob Poly, a=' + str(alpha) + ", log")
# #     #         plt.plot(mgsa_prob_rbf_num_of_iter[path][alpha], mgsa_prob_rbf_hw[path][alpha], label = 'MGSA Prob Rbf, a=' + str(alpha) + ", log")
# #     else:
# #         for alpha in mgsa_prob_num_of_iter[path]:
# #             suffix = ""
# #             if path.endswith('/adj_weights'):
# #                 suffix = ", adj_weights"
# #             elif path.endswith('/not_adj_weights'):
# #                 suffix = ", not_adj_weights"
# #             # plt.plot(mgsa_prob_num_of_iter[path][alpha], mgsa_prob_hw[path][alpha], label = 'MGSA Prob Linear, a=' + str(alpha) + suffix)
# #             # plt.plot(mgsa_prob_poly_num_of_iter[path][alpha], mgsa_prob_poly_hw[path][alpha], label = 'MGSA Prob Poly, a=' + str(alpha) + suffix)
# #             plt.plot(mgsa_prob_rbf_num_of_iter[path][alpha], mgsa_prob_rbf_hw[path][alpha], label = 'MGSA Prob Rbf, a=' + str(alpha) + suffix)

# # plt.legend()
# # plt.show()

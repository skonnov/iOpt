from problems.grishagin_mco import Grishagin_mco
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.models.model import Model
from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from iOpt.models.model_svm_proba_adj_weights import ModelLinearSVCprobaAdjWeights, ModelPolySVCprobaAdjWeights, ModelRbfSVCprobaAdjWeights
from iOpt.models.model_svm_proba_log_normalized import ModelLinearSVCprobaLogNorm, ModelPolySVCprobaLogNorm, ModelRbfSVCprobaLogNorm
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane
# from iOpt.models.model_xgboost import ModelXGBoostProba
from iOpt.models.model_nn import ModelNNProba
from iOpt.models.model_nn_1_layer import ModelNNProba1Layer
# from iOpt.models.model_random_forest import ModelRandomForestProba
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
import time
import os

from datetime import datetime


def calculate_grishagin_mco(func_ids, alpha: float = 0., model: Model = None, iters_limit = None, eps = None, async_scheme=False, parallel_dots_count=None):
    if iters_limit is None:
        iters_limit = 16000
    if eps is None:
        eps = 0.01
    problem = Grishagin_mco(len(func_ids), [func for func in func_ids])

    params = SolverParameters(r=2.5, eps=eps, iters_limit=iters_limit,
                              number_of_lambdas=50, start_lambdas=[[i % 2 for i in range(len(func_ids))]],
                              is_scaling=False, number_of_parallel_points=parallel_dots_count,
                              async_scheme=async_scheme, alpha=alpha)

    solver = Solver(problem=problem, parameters=params, model=model)

    sol = solver.solve()

    # output of the Pareto set (coordinates - function values)
    val = [[trial.function_values[i].value for i in range(len(func_ids))] for trial in sol.best_trials]
    hw = pg.hypervolume(val)
    hw_dot = [2. for i in range(len(func_ids))]
    hw_index = hw.compute(hw_dot)
    # draw(solver, model)
    return (hw_index, solver.method.iterations_count)

def draw(solver: Solver, model: Model = None):
    ax = plt.gca()
    dots = [(trial, 1) for trial in solver.search_data.solution.best_trials]
    for dot in solver.search_data:
        is_best_dot = False
        for best_dot in solver.search_data.solution.best_trials:
            if np.linalg.norm(dot.point.float_variables - best_dot.point.float_variables) < 1e-5:
                is_best_dot = True
                break
        if not is_best_dot:
            dots.append((dot, 0))
    fit_data = np.array([[func_value.value for func_value in dot.function_values] for (dot, _) in dots])
    fit_data_class = np.array([dot_class for (_, dot_class) in dots])

    if model:
        DecisionBoundaryDisplay.from_estimator(
            model.get_model(),
            fit_data,
            plot_method="contour",
            colors="k",
            levels=[0],
            alpha=0.5,
            linestyles=["-"],
            ax=ax,
        )

    # TMP: draw plt with all dots and linear regression function
    plt.scatter(fit_data[:, 0], fit_data[:, 1], c=fit_data_class, s=30, cmap=plt.cm.Paired)
    plt.show()

def solve(filename, target_eps_arr, alpha_arr, models, func_ids_arr, async_scheme, parallel_dots_count):
    with open(filename, "w") as f:
        print("f = functions ids, a = alpha, hw = hw_index, n = number of iterations, e = target accuracy, m = model name, p_dots = parallel_dots, async = is async", file=f)
        for func_ids in func_ids_arr:
            print("f", func_ids, file=f)
            print("f", func_ids)
            for target_eps in target_eps_arr:
                print("e", target_eps, file=f)
                print("e", target_eps)
                for model in models:
                    if model is None:
                        print("m", "mgsa", file=f)
                        print("m", "mgsa")
                    else:
                        if model.name().startswith("NN"):
                            print("m", model.name() + "_" + str(model.weights[0]) + "_" + str(model.weights[1]), file=f)
                            print("m", model.name() + "_" + str(model.weights[0]) + "_" + str(model.weights[1]))
                            # if is_cuda:
                            #     device = torch.device("cuda")
                            #     print(f"Using device: {device}")
                            # else:
                            #     device = torch.device("cpu")
                            #     print(f"Using device: {device}")
                        else:
                            print("m", model.name(), file=f)
                            print("m", model.name())
                    print("p_dots", parallel_dots_count, file=f)
                    print("p_dots", parallel_dots_count)
                    print("async", async_scheme, file=f)
                    print("async", async_scheme)

                    for alpha in alpha_arr:
                        print("a", alpha, file=f)
                        print("a", alpha)
                        time1 = time.time()

                        hw_index, iter_count = calculate_grishagin_mco(func_ids, alpha=alpha, model=model, eps=target_eps, parallel_dots_count=parallel_dots_count,
                                                                        async_scheme=async_scheme)
                        time2 = time.time()
                        print("hw", hw_index, "n", iter_count)
                        print("hw", hw_index, "n", iter_count, file=f)
                        print("t", time2 - time1)
                        print("t", time2 - time1, file=f)
                        f.flush()

if __name__ == "__main__":
    # generate 100 pairs of grishagin problem
    func_ids = []
    for i in range(1, 21):
        func_ids.append([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])

    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    func_ids_local = np.array_split(np.array(func_ids), task_count)[task_id]

    time1 = time.time()
    for (async_scheme, dots_count) in [(False, 1), (False, 2), (True, 2), (True, 4), (True, 8), (True, 16), (True, 20)]:
        solve("mso_grishagin_4_crit_mgsa_nn_2_layers_" + dt_string + "_task_" + str(task_id) + "_" + str(async_scheme) +"_" + str(dots_count) +".txt",    (0.1, 0.05, 0.01), [0.03],
              [ModelNNProba([0.2, 0.8], 4), ModelNNProba([0.25, 0.75], 4), ModelNNProba([0.3, 0.7], 4)], func_ids_local, async_scheme, dots_count)
        solve("mso_grishagin_4_crit_mgsa_nn_1_layer_" + dt_string + "_task_" + str(task_id) + "_" + str(async_scheme) +"_" + str(dots_count) +".txt",    (0.1, 0.05, 0.01), [0.05],
              [ModelNNProba1Layer([0.2, 0.8], 4)], func_ids_local, async_scheme, dots_count)
        solve("mso_grishagin_4_crit_" + dt_string + "_task_" + str(task_id) + "_" + str(async_scheme) +"_" + str(dots_count) +".txt",         (0.1, 0.05, 0.01), [0.],
              [None], func_ids_local, async_scheme, dots_count)
        solve("mso_grishagin_4_crit_dist_" + dt_string + "_task_" + str(task_id) + "_" + str(async_scheme) +"_" + str(dots_count) +".txt",    (0.1, 0.05, 0.01), [0.01],
              [ModelLinearSVChyperplane()], func_ids_local, async_scheme, dots_count)

        solve("mso_grishagin_4_crit_log_norm_" + dt_string + "_task_" + str(task_id) + "_" + str(async_scheme) +"_" + str(dots_count) +".txt", (0.1, 0.05, 0.01), [0.08],
                                    (ModelLinearSVCprobaLogNorm(), ModelPolySVCprobaLogNorm(), ModelRbfSVCprobaLogNorm()), func_ids_local, async_scheme, dots_count)

    # solve("mso_grishagin_4_crit_not_weighted_" + dt_string + "_task_" + str(task_id) + ".txt", (0.1, 0.05, 0.01), (0.05, 0.08, 0.1), (ModelLinearSVCproba(), ModelPolySVCproba(), ModelRbfSVCproba()), func_ids_local)
    # solve("mso_grishagin_4_crit_weighted_" + dt_string + "_task_" + str(task_id) + ".txt", (0.1, 0.05, 0.01), (0.05, 0.08, 0.1),
    #                             (ModelLinearSVCprobaAdjWeights(), ModelPolySVCprobaAdjWeights(), ModelRbfSVCprobaAdjWeights()), func_ids_local)
    # solve("mso_grishagin_4_crit_mgsa_xgboost_" + dt_string + "_task_" + str(task_id) + ".txt",    (0.1, 0.05, 0.01), [0.09, 0.1, 0.15, 0.2, 0.3], [ModelXGBoostProba()], func_ids_local)

    # model_weights = [[0.2, 0.8], [0.1, 0.9], [0.05, 0.85], [0.01, 0.99]]
    # solve("mso_grishagin_4_crit_NN_" + dt_string + "_task_" + str(task_id) + ".txt", (0.1, 0.05, 0.01), [0.01, 0.03, 0.05, 0.08, 0.1, 0.5], [ModelNNProba(model_weight, 4) for model_weight in model_weights], func_ids_local)
    # time_1_2 = time.time()
    # print("Total time for the NN model spent:", time_1_2 - time1, "seconds")
    # solve("mso_grishagin_4_crit_dist_" + dt_string + "_task_" + str(task_id) + ".txt",    (0.1, 0.05, 0.01), [0.01], [ModelLinearSVChyperplane()], func_ids_local)

    time2 = time.time()
    print("Total time for the script spent:", time2 - time1, "seconds")

from problems.grishagin_mco import Grishagin_mco
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.models.model import Model
from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from iOpt.models.model_svm_proba_adj_weights import ModelLinearSVCprobaAdjWeights, ModelPolySVCprobaAdjWeights, ModelRbfSVCprobaAdjWeights
from iOpt.models.model_svm_proba_log_normalized import ModelLinearSVCprobaLogNorm, ModelPolySVCprobaLogNorm, ModelRbfSVCprobaLogNorm
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane
from iOpt.models.model_xgboost import ModelXGBoostProba
# from iOpt.models.model_random_forest import ModelRandomForestProba
# from iOpt.models.model_xgboost import ModelXGBoostProba
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
import time

from datetime import datetime

def calculate_grishagin_mco(func_ids, alpha: float = 0., model: Model = None, iters_limit = None, eps = None):
    if iters_limit is None:
        iters_limit = 16000
    if eps is None:
        eps = 0.01
    problem = Grishagin_mco(len(func_ids), func_ids)

    params = SolverParameters(r=2.5, eps=eps, iters_limit=iters_limit,
                              number_of_lambdas=50, start_lambdas=[[0, 1]],
                              is_scaling=False, number_of_parallel_points=2,
                              async_scheme=False, alpha=alpha)

    solver = Solver(problem=problem, parameters=params, model=model)

    sol = solver.solve()

    # output of the Pareto set (coordinates - function values)
    val = [[trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]

    hw = pg.hypervolume(val)
    hw_index = hw.compute([1., 1.])
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


def solve(filename, target_eps_arr, alpha_arr, models, func_ids_arr):
    with open(filename, "w") as f:
        print("f = functions ids, a = alpha, hw = hw_index, n = number of iterations, e = target accuracy, m = model name", file=f)
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
                        print("m", model.name(), file=f)
                        print("m", model.name())
                    for alpha in alpha_arr:
                        print("a", alpha, file=f)
                        print("a", alpha)
                        hw_index, iter_count = calculate_grishagin_mco(func_ids, alpha=alpha, model=model, eps=target_eps)
                        print("hw", hw_index, "n", iter_count, file=f)


if __name__ == "__main__":
    np.random.seed(42)

    # generate 100 pairs of grishagin problem
    func_ids = []
    for i in range(1, 50):
        func_ids.append((i * 2, i * 2 + 1))
    for i in range(50):
        func_ids.append((i, 99 - i))

    func_ids.append((30, 45))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = "mso_grishagin_mgsa_" + dt_string

    time1 = time.time()

    solve("mso_grishagin_mgsa_" + dt_string + ".txt",         (0.1, 0.05, 0.01), [0.],   [None], func_ids)
    solve("mso_grishagin_mgsa_dist_" + dt_string + ".txt",    (0.1, 0.05, 0.01), [0.01], [ModelLinearSVChyperplane()], func_ids)
    solve("mso_grishagin_not_weighted_" + dt_string + ".txt", [0.01], (0.03, 0.09), (ModelLinearSVCproba(), ModelPolySVCproba(), ModelRbfSVCproba()), func_ids)
    solve("mso_grishagin_weighted_" + dt_string + ".txt", (0.1, 0.05, 0.01), (0.03, 0.09),
                                (ModelLinearSVCprobaAdjWeights(), ModelPolySVCprobaAdjWeights(), ModelRbfSVCprobaAdjWeights()), func_ids)
    solve("mso_grishagin_log_norm_" + dt_string + ".txt", [0.01], (0.03, 0.09),
                                (ModelLinearSVCprobaLogNorm(), ModelPolySVCprobaLogNorm(), ModelRbfSVCprobaLogNorm()), func_ids)
    solve("mso_grishagin_mgsa_xgboost_" + dt_string + ".txt",    (0.1, 0.05, 0.01), [0.01, 0.02, 0.03, 0.04, 0.08], [ModelXGBoostProba()], func_ids)

    time2 = time.time()
    print("Total time for the script spent:", time2 - time1, "seconds")

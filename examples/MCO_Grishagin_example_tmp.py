from problems.grishagin_mco import Grishagin_mco
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.models.model import Model
from iOpt.models.model_linear_svm_proba import ModelLinearSVCproba
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane
# from iOpt.models.model_random_forest import ModelRandomForestProba
# from iOpt.models.model_xgboost import ModelXGBoostProba
from iOpt.models.model_poly_svm_proba import ModelPolySVCproba
from iOpt.models.model_rbf_svm_proba import ModelRbfSVCproba
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
import time

from datetime import datetime
# start_time = time.time()
def calculate_grishagin_mco_rep(func_id_1: int, func_id_2: int, alpha: float = 0., model: Model = None, iters_limit = None, eps = None, repeat_count = 1):
    hw_index = 0.
    iter_count = 0.
    for _ in range(repeat_count):
        cur_hw_index, cur_iter_count = calculate_grishagin_mco(func_id_1, func_id_2, alpha, model, iters_limit, eps)
        hw_index += cur_hw_index
        iter_count += cur_iter_count
    hw_index /= repeat_count
    iter_count /= repeat_count
    return hw_index, iter_count

def calculate_grishagin_mco(func_id_1: int, func_id_2: int, alpha: float = 0., model: Model = None, iters_limit = None, eps = None, name="", axes=None, ax1 = 0, ax2 = 0):
    if iters_limit is None:
        iters_limit = 16000
    if eps is None:
        eps = 0.01
    problem = Grishagin_mco(2, [func_id_1, func_id_2])

    params = SolverParameters(r=2.5, eps=eps, iters_limit=iters_limit,
                              number_of_lambdas=50, start_lambdas=[[0, 1]],
                              is_scaling=False, number_of_parallel_points=2,
                              async_scheme=True, alpha=alpha)

    solver = Solver(problem=problem, parameters=params, model=model)

    sol = solver.solve()
    # output of the Pareto set (coordinates - function values)
    val = [[trial.function_values[i].value for i in range(2)] for trial in sol.best_trials]

    hw = pg.hypervolume(val)
    hw_index = hw.compute([1., 1.])
    
    draw_heatmap(solver, model, name=name, axes=axes, ax1=ax1, ax2=ax2)
    # draw(solver, model)
    return (hw_index, solver.method.iterations_count)

def draw_heatmap(solver: Solver, model: Model, name="", axes=None, ax1=0, ax2=0):
    minn = None
    maxx = None
    for dot in solver.search_data:
        if minn is None:
            minn = [func_value.value for func_value in dot.function_values]
            maxx = [func_value.value for func_value in dot.function_values]
        else:
            minn = [min(minn[i], func_value.value) for i, func_value in enumerate(dot.function_values)]
            maxx = [max(maxx[i], func_value.value) for i, func_value in enumerate(dot.function_values)]

    minn = [round(cur_min - 1) for cur_min in minn]
    maxx = [round(cur_max + 1) for cur_max in maxx]

    print(minn, maxx)

    x1 = np.linspace(minn[0], maxx[0], 50)
    x2 = np.linspace(minn[1], maxx[1], 50)
    X, Y = np.meshgrid(x1, x2)

    # C = np.zeros_like(X)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         C[i, j] = model.calculate_dot_characteristic(X[i, j], Y[i, j])

    # C = C[:-1, :-1]
    vect_calc = np.vectorize(model.calculate_dot_characteristic)
    z = vect_calc(X, Y)
    if ax1 != 0:
        z = np.log(z)
    colormap = "viridis"

    # if ax1 == 0:
    #     colormap = "viridis_r"
    c = axes[ax1, ax2].pcolormesh(x1, x2, z, cmap=colormap)


    # c = plt.pcolormesh(x1, x2, z)
    # plt.colorbar(c)
    plt.colorbar(c, ax=axes[ax1, ax2])

    dots = [[func_value.value for func_value in trial.function_values] for trial in solver.search_data.solution.best_trials]
    dots.sort()
    axes[ax1, ax2].scatter([dot[0] for dot in dots], [dot[1] for dot in dots], color="red")
    # plt.scatter([dot[0] for dot in dots], [dot[1] for dot in dots], color="red")

    axes[ax1, ax2].set_title(name)
    # plt.title(name)
    # plt.show()

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

if __name__ == "__main__":
    # generate 100 pairs of grishagin problem
    func_ids = []
    for i in range(1, 50):
        func_ids.append((i * 2, i * 2 + 1))
    for i in range(50):
        func_ids.append((i, 99 - i))

    func_ids.append((30, 45))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = "mso_grishagin_" + dt_string
    
    filename = "tmp.txt"
    with open(filename, "w") as f:
        print("f = functions ids, a = alpha, hw = hw_index, n = number of iterations, e = target accuracy, ", end='', file=f)
        print("d_mgsa = approach without machine learning methods, ", end='', file=f)
        print("d_appr = approach with distance to the hyperplane, d_prob = approach with probabilities", file=f)
        time1 = time.time()
        for func_1, func_2 in func_ids:
            print(func_1, func_2)
            print("f", func_1, func_2, file=f)
            # approach with distance to the hypeplane
            # for target_eps in (0.1, 0.05, 0.01):
            for target_eps in [0.01]:

                fig, axes = plt.subplots(4, 2)
                fig.suptitle("funcs " + str(func_1) + " " + str(func_2) + ", e=" + str(target_eps))
                print("e", target_eps, file=f)

                # print("d_mgsa", file=f)
                # hw_index, iter_count = calculate_grishagin_mco(func_id_1=func_1, func_id_2=func_2, eps=target_eps)
                # print("hw", hw_index, "n", iter_count, file=f)
                # f.flush()

                print("d_dist")
                model = ModelLinearSVChyperplane()
                hw_index, iter_count = calculate_grishagin_mco(func_id_1=func_1, func_id_2=func_2,
                                                               alpha=0.01, model=model, eps=target_eps,
                                                               name="d_dist", axes=axes, ax1=0, ax2=0)
                axes[0][1].axis('off')
                print("hw", hw_index, "n", iter_count, file=f)
                f.flush()

                print("d_prob")
                for i, alpha in enumerate((0.03, 0.09)):
                    print("a", alpha)
                    model = ModelLinearSVCproba()
                    hw_index, iter_count = calculate_grishagin_mco(func_id_1=func_1, func_id_2=func_2,
                                                                   alpha=alpha, model=model, eps=target_eps,
                                                                   name="d_prob, alpha = " + str(alpha), axes=axes, ax1=1, ax2=i)
                    print("hw", hw_index, "n", iter_count, file=f)
                f.flush()
                print("d_prob_poly")
                for i, alpha in enumerate((0.03, 0.09)):
                    print("a", alpha)
                    model = ModelPolySVCproba()
                    hw_index, iter_count = calculate_grishagin_mco(func_id_1=func_1, func_id_2=func_2,
                                                                   alpha=alpha, model=model, eps=target_eps,
                                                                   name="d_prob_poly, alpha = " + str(alpha), axes=axes, ax1=2, ax2=i)
                    print("hw", hw_index, "n", iter_count, file=f)
                f.flush()

                print("d_prob_rbf")
                for i, alpha in enumerate((0.03, 0.09)):
                    print("a", alpha)
                    model = ModelRbfSVCproba()
                    hw_index, iter_count = calculate_grishagin_mco(func_id_1=func_1, func_id_2=func_2,
                                                                   alpha=alpha, model=model, eps=target_eps,
                                                                   name="d_prob_rbf, alpha = " + str(alpha), axes=axes, ax1=3, ax2=i)
                    print("hw", hw_index, "n", iter_count, file=f)
                f.flush()
                plt.show()
        time2 = time.time()
        print("Total time for the script spent:", time2 - time1, "seconds")
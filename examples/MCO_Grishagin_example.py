from problems.grishagin_mco import Grishagin_mco
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import pygmo as pg
import numpy as np
if __name__ == "__main__":

    problem = Grishagin_mco(2, [2, 3])

    params = SolverParameters(r=2.5, eps=0.01, iters_limit=16000,
                              number_of_lambdas=50, start_lambdas=[[0, 1]],
                              is_scaling=False, number_of_parallel_points=2,
                              async_scheme=True)

    solver = Solver(problem=problem, parameters=params, useHyperplaneCalc=False)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)

    sol = solver.solve()

    # output of the Pareto set (coordinates - function values)
    var = [trial.point.float_variables for trial in sol.best_trials]
    val = [[trial.function_values[i].value for i in range(2)]for trial in sol.best_trials ]
    print("size pareto set: ", len(var))
    for fvar, fval in zip(var, val):
        print(fvar, fval)

    hw = pg.hypervolume(val)
    hw_index = hw.compute([1., 1.])
    print("hw_index: ", hw_index)

    # x1 = [trial.point.float_variables[0] for trial in sol.best_trials]
    # x2 = [trial.point.float_variables[1] for trial in sol.best_trials]
    #
    # plt.plot(x1, x2, 'ro')
    # plt.show()
    #
    # fv1 = [trial.function_values[0].value for trial in sol.best_trials]
    # fv2 = [trial.function_values[1].value for trial in sol.best_trials]
    #
    # plt.plot(fv1, fv2, 'ro')
    # plt.show()

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

    if solver.search_data.is_hyperplane_init:
        DecisionBoundaryDisplay.from_estimator(
            solver.search_data.clf,
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
    print(len(fit_data), " <----- number of dots!")
    plt.show()

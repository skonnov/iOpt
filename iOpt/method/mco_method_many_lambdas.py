from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.mco_method import MCOMethod
from iOpt.method.mco_optim_task import MCOOptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

class MCOMethodManyLambdas(MCOMethod):
    """
    The MCOMethodManyLambdas class contains an implementation of
    the Global Search Algorithm in the case of multiple convolutions
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MCOOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator
                 ):
        super().__init__(parameters, task, evolvent, search_data, calculator)
        self.is_recalc_all_convolution = True
        self.max_iter_for_convolution = 0
        self.number_of_lambdas = parameters.number_of_lambdas

        if parameters.start_lambdas:
            self.start_lambdas = parameters.start_lambdas
        else:
            self.start_lambdas = []

        self.current_num_lambda = 0
        self.lambdas_list = []
        self.iterations_list = []

        self.convolution = task.convolution

        self.init_lambdas()

    def check_stop_condition(self) -> bool:
        if super().check_stop_condition():
            if self.current_num_lambda < self.number_of_lambdas:
                self.change_lambdas()
        return super().check_stop_condition()

    def change_lambdas(self) -> None:
        self.set_min_delta(1)
        self.current_num_lambda += 1
        if self.current_num_lambda < self.number_of_lambdas:
            self.current_lambdas = self.lambdas_list[self.current_num_lambda]
            self.task.convolution.lambda_param = self.current_lambdas

            self.iterations_list.append(self.iterations_count)
            max_iter_for_convolution = int((self.parameters.global_method_iteration_count /
                                            self.number_of_lambdas) * (self.current_num_lambda + 1))
            self.set_max_iter_for_convolution(max_iter_for_convolution)
            self.calculate_sep_hyperplane()

    def calc_distance(self, float_variables_1, float_variables_2):
        dist = 0.
        for i in range(len(float_variables_1)):
            dist += (float_variables_1[i] - float_variables_2[i]) ** 2
        dist **= 1 / 2
        return dist

    def calculate_sep_hyperplane(self) -> None:
        dots = [(trial, 1) for trial in self.search_data.solution.best_trials]
        dot_in = 0
        dot_out = 0
        for dot in self.search_data:
            is_best_dot = False
            for best_dot in self.search_data.solution.best_trials:
                if abs(self.calc_distance(dot.point.float_variables, best_dot.point.float_variables)) < 1e-5:
                    is_best_dot = True
                    dot_in += 1
                    break
            if not is_best_dot:
                dot_out += 1
                dots.append((dot, 0))
        fit_data = np.array([[func_value.value for func_value in dot.function_values] for (dot, _) in dots])
        fit_data_class = np.array([dot_class for (_, dot_class) in dots])
        # fit_data_prob = np.array([self.parameters.pareto_weight if dot_class else (1 - self.parameters.pareto_weight) for (_, dot_class) in dots])

        # clf = svm.LinearSVC(class_weight={1: 98})  # todo: use self.parameters.pareto_weight?
        # clf = svm.LinearSVC(class_weight={1: self.parameters.pareto_weight})
        # clf = svm.LinearSVC(class_weight={0: 1 - self.parameters.pareto_weight, 1: self.parameters.pareto_weight})
        # clf = svm.LinearSVC(class_weight={0: 100 * (1 - self.parameters.pareto_weight), 1: 100 * self.parameters.pareto_weight})
        self.search_data.clf.fit(fit_data, fit_data_class)

        # TODO: do not store distances in array
        d = self.search_data.clf.decision_function(fit_data)  # need to divide the function values by the norm of the weight vector (coef_) (in case of decision_function_shape=’ovo’)?
        self.search_data.d_min = min(d)
        self.search_data.d_max = max(d)

        self.search_data.is_hyperplane_init = True


        if self.current_num_lambda == self.number_of_lambdas - 1:
            ax = plt.gca()
            if self.search_data.is_hyperplane_init:
                DecisionBoundaryDisplay.from_estimator(
                    self.search_data.clf,
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

    def init_lambdas(self) -> None:
        if self.task.problem.number_of_objectives == 2:
            if self.number_of_lambdas > 1:
                h = 1.0/(self.number_of_lambdas-1)
            else:
                h = 1
            if not self.start_lambdas:
                for i in range(self.number_of_lambdas):
                    lambda_0 = i * h
                    if lambda_0 > 1:
                        lambda_0 = lambda_0 - 1
                    lambda_1 = 1 - lambda_0
                    lambdas = [lambda_0, lambda_1]
                    self.lambdas_list.append(lambdas)
            elif len(self.start_lambdas) == self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            elif len(self.start_lambdas) == 1:
                self.lambdas_list.append(self.start_lambdas[0])
                for i in range(1, self.number_of_lambdas):
                    lambda_0 = self.start_lambdas[0][0] + i*h
                    if lambda_0 > 1:
                        lambda_0 = lambda_0 - 1
                    lambda_1 = 1 - lambda_0
                    lambdas = [lambda_0, lambda_1]
                    self.lambdas_list.append(lambdas)
        else: # многомерный случай
            if len(self.start_lambdas) == self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            else:
                if self.number_of_lambdas > 1:
                    h = 1.0 / (self.number_of_lambdas - 1)
                else:
                    h = 1
                evolvent = Evolvent([0] * self.task.problem.number_of_objectives,
                                    [1] * self.task.problem.number_of_objectives,
                                    self.task.problem.number_of_objectives)

                for i in range(self.number_of_lambdas):
                    x = i * h
                    y = evolvent.get_image(x)
                    sum = 0
                    for i in range(self.task.problem.number_of_objectives):
                        sum += y[i]
                    for i in range(self.task.problem.number_of_objectives):
                        y[i] = y[i] / sum
                    lambdas = list(y)
                    self.lambdas_list.append(lambdas)

        self.current_lambdas = self.lambdas_list[0]
        self.max_iter_for_convolution = \
            int(self.parameters.global_method_iteration_count / self.number_of_lambdas)
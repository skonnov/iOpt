from typing import List
from datetime import datetime

import traceback

import numpy as np

from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.listener import Listener
from iOpt.method.method import Method
from iOpt.method.multi_objective_optim_task import MultiObjectiveOptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solution import Solution
from iOpt.solver_parametrs import SolverParameters
from iOpt.method.process import Process

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

# TODO: remove random usage?
import random
import matplotlib.pyplot as plt

class MCOProcess(Process):
    """
    Класс MCOProcess скрывает внутреннюю имплементацию класса Solver.
    """

    def __init__(self,
                 parameters: SolverParameters,
                 task: MultiObjectiveOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 method: Method,
                 listeners: List[Listener],
                 lambdas=[]
                 ):

        super().__init__(parameters, task, evolvent, search_data, method, listeners)

        self.number_of_lambdas = parameters.number_of_lambdas
        if lambdas:
            self.start_lambdas = lambdas
        elif parameters.start_lambdas:
            self.start_lambdas = parameters.start_lambdas
        else:
            self.start_lambdas = []

        self.current_num_lambda = 0
        self.lambdas_list = []  # список всех рассматриваемых
        self.iterations_list = []

        self.convolution = task.convolution
        self.task = task

        self.init_lambdas()

    def solve(self) -> Solution:
        """
        Метод позволяет решить задачу оптимизации. Остановка поиска выполняется согласно критерию,
        заданному при создании класса Solver.

        :return: Текущая оценка решения задачи оптимизации
        """

        start_time = datetime.now()

        try:
            for i in range(self.number_of_lambdas):
                while not self.method.check_stop_condition():
                    self.do_global_iteration()
                self.change_lambdas()

        except Exception:
            print('Exception was thrown')
            print(traceback.format_exc())

        if self.parameters.refine_solution:
            self.do_local_refinement(self.parameters.local_method_iteration_count)

        result = self.get_results()
        result.solving_time = (datetime.now() - start_time).total_seconds()

        for listener in self._listeners:
            status = self.method.check_stop_condition()
            listener.on_method_stop(self.search_data, self.get_results(), status)

        return result

    def change_lambdas(self) -> None:
        self.method.set_min_delta(1)
        self.current_num_lambda += 1
        if self.current_num_lambda < self.number_of_lambdas:
            self.current_lambdas = self.lambdas_list[self.current_num_lambda]
            self.task.convolution.lambda_param = self.current_lambdas

            self.iterations_list.append(self.method.iterations_count) # здесь будет накапливаться сумма итераций
            max_iter_for_convolution = int((self.parameters.global_method_iteration_count /
                                            self.number_of_lambdas) * (self.current_num_lambda + 1))
            self.method.set_max_iter_for_convolution(max_iter_for_convolution)
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
        print("DOTS IN: ", dot_in, "DOTS OUT:", dot_out, " <================|||")
        # random.shuffle(dots)
        fit_data = np.array([[func_value.value for func_value in dot.function_values] for (dot, _) in dots])
        fit_data_class = np.array([dot_class for (_, dot_class) in dots])
        fit_data_prob = np.array([self.parameters.pareto_weight if dot_class else (1 - self.parameters.pareto_weight) for (_, dot_class) in dots])

        test_size = int(len(fit_data) * 0.3)



        # data_train       = fit_data[:-test_size]
        # data_class_train = fit_data_class[:-test_size]
        # data_test        = fit_data[-test_size:]
        # # data_class_test  = fit_data_class[-test_size:]

        clf = svm.LinearSVC(class_weight={1: 98})  # todo: use self.parameters.pareto_weight?
        # clf = svm.LinearSVC(class_weight={1: self.parameters.pareto_weight})
        # clf = svm.LinearSVC(class_weight={0: 1 - self.parameters.pareto_weight, 1: self.parameters.pareto_weight})
        # clf = svm.LinearSVC(class_weight={0: 100 * (1 - self.parameters.pareto_weight), 1: 100 * self.parameters.pareto_weight})
        clf.fit(fit_data, fit_data_class)


        # W = clf.coef_[0]
        # I = clf.intercept_
        # a = -W[0] / W[1]
        # b = I[0] / W[1]

        # xs = np.arange(-12, 2, 0.01)
        # y = [a * x + b for x in xs]

        ax = plt.gca()
        DecisionBoundaryDisplay.from_estimator(
            clf,
            fit_data,
            plot_method="contour",
            colors="k",
            levels=[0],
            alpha=0.5,
            linestyles=["-"],
            ax=ax,
        )

        self.d = clf.decision_function(fit_data)  # need to divide the function values by the norm of the weight vector (coef_) (in case of decision_function_shape=’ovo’)?
        maxx = max(self.d)
        minn = min(self.d)
        self.d = [di / maxx if di > 0 else -di / minn for di in self.d]

        # TMP: draw plt with all dots and linear regression function

        plt.scatter(fit_data[:, 0], fit_data[:, 1], c=fit_data_class, s=30, cmap=plt.cm.Paired)

        # xs = np.arange(-12, 2, 0.01)
        # y = [lr.coef_[0] * x + lr.coef_[1] for x in xs]
        # plt.plot(xs, y)
        # # plt.plot([data_test[i][0] for i in range(len(data_test))], data_test_pred)

        plt.show()
        

    def init_lambdas(self) -> None:
        if self.task.problem.number_of_objectives == 2:  # двумерный случай
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
            elif len(self.start_lambdas)==self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            elif len(self.start_lambdas)==1:
                self.lambdas_list.append(self.start_lambdas[0])
                for i in range(1, self.number_of_lambdas):
                    lambda_0 = self.start_lambdas[0][0] + i*h
                    if lambda_0 > 1:
                        lambda_0 = lambda_0 - 1
                    lambda_1 = 1 - lambda_0
                    lambdas = [lambda_0, lambda_1]
                    self.lambdas_list.append(lambdas)
        else: # многомерный случай
            if len(self.start_lambdas)==self.number_of_lambdas:
                for i in range(self.number_of_lambdas):
                    self.lambdas_list.append(self.start_lambdas[i])
            else:
                if self.number_of_lambdas > 1:
                    h = 1.0 / (self.number_of_lambdas - 1)
                else:
                    h = 1
                evolvent = Evolvent([0]*self.task.problem.number_of_objectives, [1]*self.task.problem.number_of_objectives, self.task.problem.number_of_objectives)

                for i in range(self.number_of_lambdas):
                    x = i*h
                    y = evolvent.get_image(x)
                    sum = 0
                    for i in range(self.task.problem.number_of_objectives):
                        sum += y[i]
                    for i in range(self.task.problem.number_of_objectives):
                        y[i] = y[i] / sum
                    lambdas = list(y)
                    self.lambdas_list.append(lambdas)

        self.current_lambdas = self.lambdas_list[0]
        self.method.max_iter_for_convolution = int(
            self.parameters.global_method_iteration_count / self.number_of_lambdas)

    # TODO: проверить load/store


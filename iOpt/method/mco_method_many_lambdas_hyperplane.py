from iOpt.method.mco_method_many_lambdas import MCOMethodManyLambdas

import numpy as np

class MCOMethodManyLambdasHyperplane(MCOMethodManyLambdas):
    """
    The MCOMethodManyLambdasHyperplane class contains an implementation of
    the Global Search Algorithm in the case of multiple convolutions using
    method with the construction of a separating hyperplane using machine
    learning methods
    """

    def change_lambdas(self) -> None:
        super().change_lambdas()
        if self.current_num_lambda < self.number_of_lambdas:
            self.calculate_sep_hyperplane()

    def calculate_sep_hyperplane(self) -> None:
        dots = [(trial, 1) for trial in self.search_data.solution.best_trials]
        for dot in self.search_data:
            is_best_dot = False
            for best_dot in self.search_data.solution.best_trials:
                if np.linalg.norm(dot.point.float_variables - best_dot.point.float_variables) < 1e-5:
                    is_best_dot = True
                    break
            if not is_best_dot:
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

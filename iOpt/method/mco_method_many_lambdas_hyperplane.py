from iOpt.evolvent.evolvent import Evolvent
from iOpt.method.calculator import Calculator
from iOpt.method.mco_method import MCOMethod
from iOpt.method.mco_optim_task import MCOOptimizationTask
from iOpt.method.search_data import SearchData
from iOpt.solver_parametrs import SolverParameters
from iOpt.models.model import Model
from iOpt.method.mco_method_many_lambdas import MCOMethodManyLambdas
from iOpt.method.search_data import SearchDataItem


import numpy as np

class MCOMethodManyLambdasHyperplane(MCOMethodManyLambdas):
    """
    The MCOMethodManyLambdasHyperplane class contains an implementation of
    the Global Search Algorithm in the case of multiple convolutions using
    method with the construction of a separating hyperplane using machine
    learning methods
    """
    def __init__(self,
                 parameters: SolverParameters,
                 task: MCOOptimizationTask,
                 evolvent: Evolvent,
                 search_data: SearchData,
                 calculator: Calculator,
                 model: Model,
                 ):
        super().__init__(parameters, task, evolvent, search_data, calculator)
        self.model = model


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

        self.model.init_model()

        self.model.fit(fit_data.tolist(), fit_data_class.tolist())

    def calculate_global_r(self, curr_point: SearchDataItem, left_point: SearchDataItem) -> None:
        super().calculate_global_r(curr_point, left_point)
        if left_point is None:
            return None
        r_ps = self.model.calculate_r_ps(curr_point, left_point)
        curr_point.globalR += self.parameters.alpha * r_ps

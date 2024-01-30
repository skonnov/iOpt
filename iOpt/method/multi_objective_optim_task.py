from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import copy

from iOpt.method.optim_task import OptimizationTask, TypeOfCalculation
from iOpt.method.search_data import SearchDataItem
from iOpt.problem import Problem


class Convolution(ABC):
    """
    Класс Convolution является базовым классом для различных сверток.
    Предполагается, что для каждого набора lambda будет порождён отдельный объект свертки.
    Также возможен вариант, что свертка изменяется по ссылке передаваемой через конструктор класса.
    """

    def __init__(self,
                 problem: Problem,
                 lambda_param: np.ndarray(shape=(1), dtype=np.double)
                 ):
        self.problem = problem
        self.lambda_param = lambda_param
    # Свертка меняет z у SearchDataItem. Z используется в методе для вычисления характеристик
    @abstractmethod
    def calculate_convolution(self,
                              data_item: SearchDataItem,
                              min_value: np.ndarray(shape=(1), dtype=np.double) = [],
                              max_value: np.ndarray(shape=(1), dtype=np.double) = []
                              ) -> SearchDataItem:
        pass


class MinMaxConvolution(Convolution):
    """
    """

    def __init__(self,
                 problem: Problem,
                 lambda_param: np.ndarray(shape=(1), dtype=np.double)
                 ):
        super().__init__(problem, lambda_param)
        self.is_scaling: bool = False  # от чего он будет приходящим параметром?
        print("lambda in init", self.lambda_param)


    # Свертка меняет z у SearchDataItem. Z используется в методе для вычисления характеристик
    def calculate_convolution(self,
                              data_item: SearchDataItem,
                              min_value: np.ndarray(shape=(1), dtype=np.double) = [],
                              max_value: np.ndarray(shape=(1), dtype=np.double) = []
                              ) -> SearchDataItem:

        value = 0
        print("lambda", self.lambda_param)


        for i in range(0, self.problem.number_of_objectives):
            dx = 1
            if self.is_scaling:
                dx = np.double(max_value[i] - min_value[i])
                if dx < 1e-6:
                    dx = 1

            print("data_item.function_values[i].value", data_item.function_values[i].value)
            f_value = (data_item.function_values[i].value - min_value[i]) / dx
            print("f_value", f_value)
            print("value before", value)
            value = max(value, f_value * self.lambda_param[i])
            print("value", value)

        data_item.set_z(value)
        print("data_item.set_z(value)", data_item.get_z())
        return data_item

class MultiObjectiveOptimizationTask(OptimizationTask):
    def __init__(self,
                 problem: Problem,
                 convolution: Convolution,
                 perm: np.ndarray(shape=(1), dtype=int) = None
                 ):
        super().__init__(problem, perm)
        self.convolution = convolution
        self.min_value = np.ndarray(shape=(problem.number_of_objectives,), dtype=np.double)
        self.min_value.fill(0)
        self.max_value = np.ndarray(shape=(problem.number_of_objectives,), dtype=np.double)
        self.max_value.fill(0)
        if self.problem.known_optimum:
            self.min_value = copy.deepcopy(self.problem.known_optimum[0].function_values)
            for know_optimum in self.problem.known_optimum:
                for i in range(0, self.problem.number_of_objectives):
                    if self.min_value[i] > know_optimum.function_values[i]:
                        self.min_value[i] = know_optimum.function_values[i]

    def calculate(self,
                  data_item: SearchDataItem,
                  function_index: int,
                  calculation_type: TypeOfCalculation = TypeOfCalculation.FUNCTION
                  ) -> SearchDataItem:
        """Compute selected function by number."""
        if calculation_type == TypeOfCalculation.FUNCTION:
            data_item.function_values[self.perm[function_index]] = self.problem.calculate(data_item.point,
                                                                                          data_item.function_values[
                                                                                              self.perm[
                                                                                                  function_index]])
            if not np.isfinite(data_item.function_values[self.perm[function_index]].value):
                raise Exception("Infinity values")

        elif calculation_type == TypeOfCalculation.CONVOLUTION:
            data_item = self.convolution.calculate_convolution(data_item, self.min_value, self.max_value)

        return data_item

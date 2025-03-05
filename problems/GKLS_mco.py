from iOpt.problem import Problem
from problems.GKLS_function.gkls_function import GKLSClass, GKLSFuncionType, GKLSFunction
from iOpt.trial import Point, FunctionValue, Trial
import numpy as np

class GKLS_mco(Problem):
    """
    GKLS-generator, allows to generate multi-extremal optimization problems with known properties in advance:
    The number of local minima, the sizes of their regions of attraction, the point of global minimum,
    the value of function in it, etc.
    """

    def __init__(self, count_functions: int,
                 function_numbers: np.ndarray(shape=(1), dtype=int) = None,  dimension: int = 2) -> None:
        """
        Constructor of the GKLS generator class

        :param dimension: Task dimensionality, :math:`2 <= dimension <= 5`
        :param functionNumber: set task number, :math:`1 <= functionNumber <= 100`
        """
        super(GKLS_mco, self).__init__()
        self.dimension = dimension
        self.name = "GKLS"
        self.number_of_float_variables = dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = count_functions
        self.number_of_constraints = 0

        self.float_variable_names = [str(x) for x in range(self.dimension)]

        self.lower_bound_of_float_variables = dimension * [-1]
        self.upper_bound_of_float_variables = dimension * [1]

        self.mMaxDimension: int = 5
        self.mMinDimension: int = 2

        self.count_functions = count_functions
        self.function_numbers = np.ndarray(shape=(self.count_functions, ), dtype=int)
        if len(function_numbers):
            self.function_numbers = function_numbers #сюда бы проверки всякие запихнуть
        else:
            for i in range(count_functions):
                self.function_numbers[i]=i+1 # мб добавить рандомное заполнение?

        self.num_minima: int = 50
        self.problem_class: int = GKLSClass.Simple
        self.function_class: int = GKLSFuncionType.TD

        self.functions = np.ndarray(shape=(self.count_functions,), dtype=GKLSFunction)

        for i in range(count_functions):
            self.functions[i]: GKLSFunction = GKLSFunction()

            self.functions[i].GKLS_global_value: float = -1.0
            self.functions[i].NumberOfLocalMinima: int = self.num_minima
            self.functions[i].SetDimension(self.dimension)
            self.functions[i].mFunctionType: int = self.function_class

            self.functions[i].SetFunctionClass(self.problem_class, self.dimension)

            self.functions[i].GKLS_parameters_check()
            self.functions[i].SetFunctionNumber(self.function_numbers[i])

        self.global_dist: float = self.functions[0].GKLS_global_dist
        self.global_radius: float = self.functions[0].GKLS_global_radius


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Calculate the value of a function at a given point

        :param point: coordinates of the trial point where the value of the function will be calculated.
        :param function_value: object defining the function number in the task and storing the function value.

        :return: Calculated value of the function at the point.
        """
        function_value.value = self.functions[function_value.functionID].Calculate(point.float_variables)
        return function_value

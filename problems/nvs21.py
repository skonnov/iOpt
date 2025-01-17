import numpy as np
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.trial import FunctionType
from iOpt.trial import Trial
from iOpt.problem import Problem
import math


class nvs21(Problem):
    """

    """

    def __init__(self):
        """
        Конструктор класса nvs21 problem.
        """
        super(nvs21, self).__init__()
        self.name = "nvs21"
        self.dimension = 3
        self.number_of_float_variables = 1
        self.number_of_discrete_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 2

        self.float_variable_names = np.ndarray(shape=(self.number_of_float_variables,), dtype=object)
        for i in range(self.number_of_float_variables):
            self.float_variable_names[i] = str(i)

        self.discrete_variable_names = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        for i in range(self.number_of_discrete_variables):
            self.discrete_variable_names[i] = str(i)

        self.lower_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.lower_bound_of_float_variables[0] = 0.1
        self.upper_bound_of_float_variables = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        self.upper_bound_of_float_variables[0] = 0.2

        self.discrete_variable_values = [[str(i) for i in range(1, 201)] for i in range(self.number_of_discrete_variables)]

        self.known_optimum = np.ndarray(shape=(1,), dtype=Trial)

        pointfv = np.ndarray(shape=(self.number_of_float_variables,), dtype=np.double)
        pointfv = [0.1]

        pointdv = np.ndarray(shape=(self.number_of_discrete_variables,), dtype=object)
        pointdv.fill("1")

        KOpoint = Point(pointfv, pointdv)
        KOfunV = np.ndarray(shape=(1,), dtype=FunctionValue)
        KOfunV[0] = FunctionValue()
        KOfunV[0].value = 0.000636
        self.known_optimum[0] = Trial(KOpoint, KOfunV)


    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        """
        Вычисление значения выбранной функции в заданной точке.

        :param point: координаты точки испытания, в которой будет вычислено значение функции
        :param function_value: объект определяющий номер функции в задаче и хранящий значение функции
        :return: Вычисленное значение функции в точке point
        """
        result: np.double = 0
        x = point.float_variables
        b = point.discrete_variables

        if function_value.type == FunctionType.OBJECTIV:
            result = np.double(0.00201 * pow(int(b[0]), 4) * int(b[1]) * math.sqrt(x[0]))
        elif function_value.functionID == 0:  # constraint 1
            result = np.double(-(-math.sqrt(int(b[0])) * int(b[1]) + 675.0))
        elif function_value.functionID == 1:  # constraint 2
            result = np.double(-(-0.1 * math.sqrt(int(b[0])) * math.sqrt(x[0]) + 0.419))

        function_value.value = result
        return function_value

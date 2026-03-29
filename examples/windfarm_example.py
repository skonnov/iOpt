from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from problems.windfarm import Windfarm

import numpy as np
import os


if __name__ == "__main__":
    windturbines_count = 8

    problem = Windfarm(windturbines_count)
    # method_params = SolverParameters(r=2.5, eps=0.01, start_lambdas=[[0, 1]], iters_limit=400, alpha=0.4)
    method_params = SolverParameters(r=2.5, eps=0.01, start_lambdas=[[0, 1]], alpha=0.4, iters_limit=20000)
    solver = Solver(problem, parameters=method_params)

    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()

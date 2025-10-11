from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.CNN.Problems import XGBoost_Pascal_VOC


from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from iOpt.models.model_svm_proba_adj_weights import ModelLinearSVCprobaAdjWeights, ModelPolySVCprobaAdjWeights, ModelRbfSVCprobaAdjWeights
from iOpt.models.model_svm_proba_log_normalized import ModelLinearSVCprobaLogNorm, ModelPolySVCprobaLogNorm, ModelRbfSVCprobaLogNorm
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane
from iOpt.models.model_nn import ModelNNProba

import numpy as np
import os

def f1():
    # learning_rate = {'low': 0.01, 'up': 0.1}
    num_estimators = {'low': 20, 'up': 300}
    max_depth = {'low': 2, 'up': 10.9}
    # problem = XGBoost_Pascal_VOC.XGBoostPascalVoc(learning_rate, max_depth)
    problem = XGBoost_Pascal_VOC.XGBoostPascalVoc(num_estimators, max_depth)
    method_params = SolverParameters(r=2.5, eps=0.01, start_lambdas=[[0, 1]], iters_limit=400, alpha=0.4)

    task_id=0
    task_count=1
    # task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    # task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

    # models = [None, ModelLinearSVCproba(), ModelPolySVCproba(), ModelNNProba([0.2, 0.8], 2), ModelRbfSVCproba(), ModelNNProba([0.1, 0.9], 2)]
#     models = [None, ModelLinearSVCproba(), ModelPolySVCproba(), ModelRbfSVCproba()]
    models = [ModelLinearSVChyperplane()]
    models_local = np.array_split(np.array(models), task_count)[task_id]
    for model in models_local:
        if model == None:
            solver = Solver(problem, parameters=method_params)
            print("Starting calculation without model! ")
        else:
            print("Starting calculation with model ", model.name)
            solver = Solver(problem, parameters=method_params, model=model)
        cfol = ConsoleOutputListener(mode='full')
        solver.add_listener(cfol)
        solver_info = solver.solve()

if __name__ == "__main__":
    f1()


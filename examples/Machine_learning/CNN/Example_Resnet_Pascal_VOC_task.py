from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.CNN.Problems import ResNet_Pascal_VOC


from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from iOpt.models.model_svm_proba_adj_weights import ModelLinearSVCprobaAdjWeights, ModelPolySVCprobaAdjWeights, ModelRbfSVCprobaAdjWeights
from iOpt.models.model_svm_proba_log_normalized import ModelLinearSVCprobaLogNorm, ModelPolySVCprobaLogNorm, ModelRbfSVCprobaLogNorm
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane
from iOpt.models.model_nn import ModelNNProba

import numpy as np
import os

if __name__ == "__main__":
    learning_rate = {'low': 0.0005, 'up': 0.001}

    batch_size = {'low': 1., 'up': 5.9}
    problem = ResNet_Pascal_VOC.ResNetPascalVoc(learning_rate, batch_size)
    method_params = SolverParameters(r=2.5, eps=0.01, start_lambdas=[[0, 1]], iters_limit=400, alpha=0.4)


    # task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    # task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    task_id = 0
    task_count=1

    models = [None, ModelLinearSVCproba(), ModelPolySVCproba(), ModelNNProba([0.2, 0.8], 2), ModelRbfSVCproba(), ModelNNProba([0.1, 0.9], 2)]
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


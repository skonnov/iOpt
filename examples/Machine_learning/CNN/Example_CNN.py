from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from examples.Machine_learning.CNN.Problems import CNN


from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from iOpt.models.model_svm_proba_adj_weights import ModelLinearSVCprobaAdjWeights, ModelPolySVCprobaAdjWeights, ModelRbfSVCprobaAdjWeights
from iOpt.models.model_svm_proba_log_normalized import ModelLinearSVCprobaLogNorm, ModelPolySVCprobaLogNorm, ModelRbfSVCprobaLogNorm
from iOpt.models.model_linear_svm_hyperplane import ModelLinearSVChyperplane

if __name__ == "__main__":
    learning_rate = {'low': 0.001, 'up': 0.1}
    kernel_coefficient_bound = {'low': 3, 'up': 5.01}
    problem = CNN.CNN(learning_rate, kernel_coefficient_bound)
    method_params = SolverParameters(r=2.5, eps=0.01, start_lambdas=[[0, 1]], iters_limit=400, alpha=0.4, async_scheme=True)
    solver = Solver(problem, parameters=method_params, model=ModelLinearSVCproba())
    cfol = ConsoleOutputListener(mode='full')
    solver.add_listener(cfol)
    solver_info = solver.solve()


from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.models.model_svm_proba import ModelLinearSVCproba, ModelPolySVCproba, ModelRbfSVCproba
from problems.car import Car, Car2Crits
from iOpt.method.local_optimizer import local_optimize
from iOpt.method.optim_task import OptimizationTask
from iOpt.trial import Point, Trial, FunctionValue, FunctionType
import numpy as np
import os
from iOpt.method.search_data import SearchData, SearchDataItem
# import pygmo as pg
import csv, json
from datetime import datetime
import time

if __name__ == "__main__":

    # problem = Car()
    # method_params = SolverParameters(r=4, eps=0.01, start_lambdas=[[i % 2 for i in range(3)]], alpha=0.1, iters_limit=200)

    # problem = Car2Crits()
    # method_params = SolverParameters(r=4, eps=0.001, start_lambdas=[[0, 1]], alpha=0.07, iters_limit=1000)

    # solver = Solver(problem, parameters=method_params, model=ModelRbfSVCproba())
    # # solver = Solver(problem, parameters=method_params)

    # cfol = ConsoleOutputListener(mode='full')
    # solver.add_listener(cfol)

    # sol = solver.solve()

    # val = [[trial.function_values[i].value for i in range(len(trial.function_values))] for trial in sol.best_trials]
    # hw = pg.hypervolume(val)
    # hw_dot = [1.5, 0.006]
    # hw_index = hw.compute(hw_dot)
    # print("hw index: ", hw_index)

    hw_thresholds = [3, 0.01, 0.01]
    # alphas = np.arange(0.15, 0.15 + 1e-12, 0.02)
    alpha = 0.15
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    models = [None, ModelRbfSVCproba()]

    # task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    # task_count = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    task_id = 0
    task_count = 1
    models_local = np.array_split(np.array(models), task_count)[task_id]

    output_file = f"hw_results_{dt_string}_{task_id}.csv"
    time_global_start = time.time()
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "alpha", "eps", "parallel dots", "iterations","val"])

        for eps in [0.00001]:
            for model in models_local:
                model_name = "no model"
                if model:
                    model_name = model.name()
                #for dots_count in [1, 2, 2, 4, 8, 16, 20]:
                for dots_count in [1]:
                    time_start = time.time()
                    print(f"\n=== model = {model_name}, eps = {eps}, dots count = {dots_count} ===")
                    problem = Car()
                    method_params = SolverParameters(r=4,
                                                    eps=eps,
                                                    start_lambdas=[[i % 2 for i in range(3)]],
                                                    alpha=alpha,
                                                    iters_limit=1000,
                                                    number_of_parallel_points=dots_count)
                    solver = Solver(problem, parameters=method_params, model=model)

                    cfol = ConsoleOutputListener(mode='full')
                    solver.add_listener(cfol)

                    sol = solver.solve()
                    iter_count = solver.method.iterations_count

                    val = [
                        [trial.function_values[i].value for i in range(len(trial.function_values))]
                        for trial in sol.best_trials
                    ]

                    if not val:
                        print("No best_trials found, skipping...")
                        writer.writerow([model_name, float(alpha), eps, dots_count, iter_count, json.dumps([])])
                        continue

                    print("val:", val)

                    writer.writerow([
                        model_name,
                        float(alpha),
                        float(eps),
                        int(dots_count),
                        int(iter_count),
                        json.dumps(val)
                    ])
                    f.flush()
                    time_end = time.time()
                    print("time spent for current iteration:", round(time_end - time_start, 4), "seconds")
    time_global_end = time.time()
    print(f"\nResults saved to: {os.path.abspath(output_file)}")
    print("Total time for the script spent:", round(time_global_end - time_global_start, 4), "seconds")

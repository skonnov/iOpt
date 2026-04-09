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
import pygmo as pg
import csv, json
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
    alphas = np.arange(0.15, 0.15 + 1e-12, 0.02)

    output_file = "hw_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "hw_index", "hw_dot", "val"])

        for alpha in alphas:
            print(f"\n=== alpha = {alpha:.2f} ===")


            problem = Car()
            method_params = SolverParameters(r=4, eps=0.1, start_lambdas=[[i % 2 for i in range(3)]], alpha=0.1, iters_limit=20000)
            solver = Solver(problem, parameters=method_params, model=ModelRbfSVCproba())

            cfol = ConsoleOutputListener(mode='full')
            solver.add_listener(cfol)

            sol = solver.solve()

            val = [
                [trial.function_values[i].value for i in range(len(trial.function_values))]
                for trial in sol.best_trials
            ]

            if not val:
                print("No best_trials found, skipping...")
                writer.writerow([float(alpha), None, None, json.dumps([])])
                continue

            point_dim = len(val[0])
            if len(hw_thresholds) < point_dim:
                raise ValueError(
                    f"hw_thresholds has length {len(hw_thresholds)}, "
                    f"but hypervolume points have dimension {point_dim}"
                )

            hw_dot = hw_thresholds[:point_dim]

            hw = pg.hypervolume(val)
            hw_index = hw.compute(hw_dot)

            print("val:", val)
            print("hw_dot:", hw_dot)
            print("hw index:", hw_index)

            writer.writerow([
                float(alpha),
                float(hw_index),
                # json.dumps(hw_dot),
                json.dumps(val)
            ])

    print(f"\nResults saved to: {os.path.abspath(output_file)}")

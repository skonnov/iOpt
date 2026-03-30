import numpy as np
import time
import xgboost as xgb
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import List, Dict, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import models
# from thop import profile
from dataclasses import dataclass

Vec2 = Tuple[float, float]

# -------------------- Wake + Power model (Jensen-like) --------------------

@dataclass(frozen=True)
class TurbineSpec:
    rotor_diameter_m: float          # D
    hub_height_m: float              # z
    z0_m: float                      # roughness
    ct: float                        # thrust coefficient (assumed constant)
    rho_kg_m3: float = 1.225
    cp: float = 0.45                 # simplified constant Cp

    cut_in_mps: Optional[float] = 3.0
    rated_mps: Optional[float] = 12.0
    cut_out_mps: Optional[float] = 25.0
    rated_power_w: Optional[float] = 3_000_000.0  # optional cap


def ct_to_a(ct: float) -> float:
    # CT = 4 a (1-a) -> a = (1 - sqrt(1-CT))/2
    return (1.0 - np.sqrt(max(0.0, 1.0 - ct))) / 2.0


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    """Overlap area of two circles radii r1,r2 with center distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2

    a1 = np.acos((d*d + r1*r1 - r2*r2) / (2.0 * d * r1))
    a2 = np.acos((d*d + r2*r2 - r1*r1) / (2.0 * d * r2))
    term = max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return r1*r1 * a1 + r2*r2 * a2 - 0.5 * np.sqrt(term)


def turbine_power_w(u: float, spec: TurbineSpec) -> float:
    """Cubic power with cut-in/out and rated cap."""
    if spec.cut_in_mps is not None and u < spec.cut_in_mps:
        return 0.0
    if spec.cut_out_mps is not None and u > spec.cut_out_mps:
        return 0.0

    rR = 0.5 * spec.rotor_diameter_m
    A = np.pi * rR * rR
    p = 0.5 * spec.rho_kg_m3 * A * spec.cp * (u ** 3)

    if spec.rated_power_w is not None:
        return min(p, spec.rated_power_w)
    return p


def min_distance_ok(coords_m: Sequence[Vec2], min_dist_m: float) -> bool:
    n = len(coords_m)
    for i in range(n):
        xi, yi = coords_m[i]
        for j in range(i + 1, n):
            xj, yj = coords_m[j]
            if np.hypot(xi - xj, yi - yj) < min_dist_m:
                return False
    return True


def compute_powers_jensen_2d(
    coords_m: Sequence[Vec2],
    u0_mps: float,
    wind_dir_deg: float,   # 0° -> flow to +x
    spec: TurbineSpec,
    use_overlap: bool = True,
) -> List[float]:
    """Returns per-turbine power (W) for the given layout."""
    n = len(coords_m)
    if n == 0:
        return []

    D = spec.rotor_diameter_m
    rR = 0.5 * D
    Arotor = np.pi * rR * rR

    a = ct_to_a(spec.ct)
    alpha = 0.5 / np.log(spec.hub_height_m / spec.z0_m)
    r1 = rR * np.sqrt((1.0 - a) / (1.0 - 2.0 * a))

    th = np.radians(wind_dir_deg)
    ex, ey = np.cos(th), np.sin(th)

    powers: List[float] = []
    for j in range(n):
        xj, yj = coords_m[j]
        deficits_sq = 0.0

        for i in range(n):
            if i == j:
                continue
            xi, yi = coords_m[i]
            dx, dy = (xj - xi), (yj - yi)

            # along-wind distance s and cross-wind offset d
            s = dx * ex + dy * ey
            if s <= 0.0:
                continue

            cx = dx - s * ex
            cy = dy - s * ey
            d = np.hypot(cx, cy)

            rw = r1 + alpha * s
            if d >= rw + rR:
                continue

            delta = (2.0 * a) / (1.0 + alpha * (s / r1)) ** 2

            if use_overlap:
                Aov = circle_intersection_area(rw, rR, d)
                f = Aov / Arotor
                delta_eff = f * delta
            else:
                delta_eff = delta if d <= rw else 0.0

            deficits_sq += delta_eff * delta_eff

        delta_tot = np.sqrt(deficits_sq)
        uj = max(0.0, u0_mps * (1.0 - delta_tot))
        powers.append(turbine_power_w(uj, spec))

    return powers


def mean_distance_to_center(coords_m: Sequence[Vec2], cx: float, cy: float) -> float:
    return sum(np.hypot(x - cx, y - cy) for (x, y) in coords_m) / max(1, len(coords_m))


# -------------------- Optuna multi-objective optimization --------------------

def coords_in_D_to_m(coords_in_D: Sequence[Vec2], D_m: float) -> List[Vec2]:
    """If coords are given in multiples of D (0..10), convert to meters."""
    return [(x * D_m, y * D_m) for (x, y) in coords_in_D]

class Windfarm(Problem):
    def __init__(self, windturbines_count: 10):
        super(Windfarm, self).__init__()

        self.name = "Windfarm"
        self.dimension = windturbines_count * 2 + 1 #  2 coords for each wind turbine + wind
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        # self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.float_variable_names = np.ndarray(shape=(self.dimension,), dtype=object)

        for i in range(windturbines_count):
            self.float_variable_names[i * 2] = str(f"turbine_{i}_x")
            self.float_variable_names[i * 2 + 1] = str(f"turbine_{i}_y")
        self.float_variable_names[2 * windturbines_count] = str("wind degree")

        self.lower_bound_of_float_variables = np.ndarray(shape=(windturbines_count * 2 + 1), dtype=np.double)
        self.lower_bound_of_float_variables.fill(0)
        self.upper_bound_of_float_variables = np.ndarray(shape=(windturbines_count * 2 + 1), dtype=np.double)
        self.upper_bound_of_float_variables.fill(10)
        self.upper_bound_of_float_variables[-1] = 359

        self.windturbines_count = windturbines_count
        self.spec = TurbineSpec(
        rotor_diameter_m=120.0,  # D
        hub_height_m=100.0,
        z0_m=0.1,
        ct=0.8,
        cp=0.45,
        rated_power_w=3_000_000.0,  # 3 MW cap (optional)
    )

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        coords_D = []


        center_x = float(5 * self.spec.rotor_diameter_m) + 5
        center_y = float(5 * self.spec.rotor_diameter_m) + 5
        # center_x = 5
        # center_y = 5


        for i in range(self.windturbines_count):
            # coords_D.append((np.round(point.float_variables[2 * i]), np.round(point.float_variables[2 * i + 1])))
            coords_D.append((point.float_variables[2 * i], point.float_variables[2 * i + 1]))

        coords_m = coords_in_D_to_m(coords_D, self.spec.rotor_diameter_m)
        wind_degree = point.float_variables[-1]
        print(coords_D, wind_degree)
        # P = compute_powers_jensen_2d_coords_m(
        #     coords_m=coords_m,
        #     u0_mps=wind_speed,
        #     wind_dir_deg=0.0,  # wind flows to +x (from left to right)
        #     spec=self.spec,
        #     use_overlap=True,
        # )



        powers = compute_powers_jensen_2d(coords_m, u0_mps=10.0, wind_dir_deg=wind_degree, spec=self.spec, use_overlap=True)
        total_power = sum(powers)
        avg_dist = mean_distance_to_center(coords_m, center_x, center_y)


        min_dist_m = 1.5 * self.spec.rotor_diameter_m
        if not min_distance_ok(coords_m, min_dist_m):
            function_values[0].value = 1e15
            function_values[1].value = avg_dist
            return function_values


        function_values[0].value = -total_power / 1e6
        function_values[1].value = avg_dist / self.spec.rotor_diameter_m
        return function_values

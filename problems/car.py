from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem

@dataclass
class QuarterCarBase:
    ms: float = 325.0      # подрессоренная масса, кг
    mu: float = 45.0       # неподрессоренная масса, кг
    sim_time: float = 4.0
    n_points: int = 2000


def half_cosine_bump(t: np.ndarray, t0: float, duration: float, height: float) -> np.ndarray:
    y = np.zeros_like(t, dtype=float)
    mask = (t >= t0) & (t <= t0 + duration)
    tau = (t[mask] - t0) / duration
    y[mask] = 0.5 * height * (1 - np.cos(2 * np.pi * tau))
    return y


def road_profile(t: np.ndarray) -> np.ndarray:
    """
    Дорожный профиль:
    - одиночная неровность,
    - затем участок волнистой дороги
    """
    t = np.asarray(t, dtype=float)

    bump = half_cosine_bump(t, t0=0.5, duration=0.18, height=0.05)

    rough = np.zeros_like(t)
    mask = (t >= 1.5) & (t <= 3.5)
    rough[mask] = 0.008 * np.sin(2 * np.pi * 7.0 * (t[mask] - 1.5))

    return bump + rough


class QuarterCarModel:
    """
    y = [zs, zs_dot, zu, zu_dot]
    zs  - перемещение кузова
    zu  - перемещение неподрессоренной массы
    zr  - профиль дороги
    """

    def __init__(self, base: QuarterCarBase, ks: float, cs: float, kt: float):
        self.base = base
        self.ks = ks
        self.cs = cs
        self.kt = kt

    def rhs(self, t: float, y):
        zs, zs_dot, zu, zu_dot = y
        zr = road_profile(np.array([t]))[0]

        suspension_force = self.ks * (zs - zu) + self.cs * (zs_dot - zu_dot)
        tire_force = self.kt * (zu - zr)

        zs_ddot = (-suspension_force) / self.base.ms
        zu_ddot = (suspension_force - tire_force) / self.base.mu

        return [zs_dot, zs_ddot, zu_dot, zu_ddot]


def simulate(base: QuarterCarBase, ks: float, cs: float, kt: float) -> dict:
    model = QuarterCarModel(base, ks=ks, cs=cs, kt=kt)

    t_eval = np.linspace(0.0, base.sim_time, base.n_points)
    y0 = np.zeros(4)

    sol = solve_ivp(
        fun=model.rhs,
        t_span=(0.0, base.sim_time),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    zs, zs_dot, zu, zu_dot = sol.y
    zr = road_profile(t)

    body_acc = np.gradient(zs_dot, t)
    tire_deflection = zu - zr
    suspension_travel = zs - zu

    comfort_rms = float(np.sqrt(np.mean(body_acc ** 2)))
    road_holding_rms = float(np.sqrt(np.mean(tire_deflection ** 2)))
    suspension_travel_rms = float(np.sqrt(np.mean(suspension_travel ** 2)))

    return {
        "t": t,
        "zs": zs,
        "zu": zu,
        "zr": zr,
        "body_acc": body_acc,
        "tire_deflection": tire_deflection,
        "suspension_travel": suspension_travel,
        "comfort_rms": comfort_rms,
        "road_holding_rms": road_holding_rms,
        "suspension_travel_rms": suspension_travel_rms,
    }


BASE = QuarterCarBase()

class Car(Problem):
    def __init__(self):
        super(Car, self).__init__()
        self.name = "Car"
        self.dimension = 3
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 3
        self.number_of_constraints = 0
        self.float_variable_names = np.array(["ks", "cs", "kt"], dtype=object)
        self.lower_bound_of_float_variables = np.array([10_000.0, 500.0, 120_000.0], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([40_000.0, 4_500.0, 260_000.0], dtype=np.double)

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        ks = point.float_variables[0]
        cs = point.float_variables[1]
        kt = point.float_variables[2]
        result = simulate(BASE, ks=ks, cs=cs, kt=kt)

        function_values[0].value = result["comfort_rms"]
        function_values[1].value = result["road_holding_rms"]
        function_values[2].value = result["suspension_travel_rms"]

        return function_values

class Car2Crits(Problem):
    def __init__(self):
        super(Car2Crits, self).__init__()
        self.name = "Car"
        self.dimension = 3
        self.number_of_float_variables = self.dimension
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.float_variable_names = np.array(["ks", "cs", "kt"], dtype=object)
        self.lower_bound_of_float_variables = np.array([10_000.0, 500.0, 120_000.0], dtype=np.double)
        self.upper_bound_of_float_variables = np.array([40_000.0, 4_500.0, 260_000.0], dtype=np.double)

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        ks = point.float_variables[0]
        cs = point.float_variables[1]
        kt = point.float_variables[2]
        result = simulate(BASE, ks=ks, cs=cs, kt=kt)

        function_values[0].value = result["comfort_rms"]
        function_values[1].value = result["suspension_travel_rms"]

        return function_values

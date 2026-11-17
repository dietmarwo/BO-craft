"""Benchmark problems with known optima, in pure Python.

Each problem gives BO-craft parameters, objectives, optional constraints, the
latent (noise-free) metric function, the optimum, and a noise scale: the
standard deviation of the objective over uniform random points, so that a noise
level of 0.1 means a tenth of the objective's typical spread. Scales were
measured on 4000 uniform points (seed 0) and rounded.
"""

import math

HARTMANN6_A = [1.0, 1.2, 3.0, 3.2]
HARTMANN6_AA = [
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
]
HARTMANN6_P = [
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
]


def hartmann6(v):
    return -sum(
        HARTMANN6_A[i]
        * math.exp(-sum(HARTMANN6_AA[i][j] * (v[j] - HARTMANN6_P[i][j] * 1e-4) ** 2 for j in range(6)))
        for i in range(4)
    )


def branin(x1, x2):
    b, c, t = 5.1 / (4 * math.pi**2), 5 / math.pi, 1 / (8 * math.pi)
    return (x2 - b * x1 * x1 + c * x1 - 6) ** 2 + 10 * (1 - t) * math.cos(x1) + 10


def currin(u1, u2):
    """Currin's function on the unit square."""
    u2 = max(u2, 1e-12)
    return (
        (1 - math.exp(-1 / (2 * u2)))
        * (2300 * u1**3 + 1900 * u1**2 + 2092 * u1 + 60)
        / (100 * u1**3 + 500 * u1**2 + 4 * u1 + 20)
    )


def rosenbrock(v):
    return sum(100 * (v[i + 1] - v[i] ** 2) ** 2 + (1 - v[i]) ** 2 for i in range(len(v) - 1))


def ackley(v):
    d = len(v)
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum(t * t for t in v) / d))
        - math.exp(sum(math.cos(2 * math.pi * t) for t in v) / d)
        + 20
        + math.e
    )


def continuous(name, lower, upper):
    return {"name": name, "type": "continuous", "lower": lower, "upper": upper}


def _branin_problem():
    return {
        "parameters": [continuous("x1", -5.0, 10.0), continuous("x2", 0.0, 15.0)],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {"y": branin(x["x1"], x["x2"])},
        "optimum": 0.397887357729738,
        "scale": 52.0,
    }


def _hartmann6_problem():
    return {
        "parameters": [continuous(f"x{i}", 0.0, 1.0) for i in range(6)],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {"y": hartmann6([x[f"x{i}"] for i in range(6)])},
        "optimum": -3.322368011391339,
        "scale": 0.38,
    }


def _branin_constrained():
    problem = _branin_problem()
    problem["constraints"] = [{"name": "g"}]
    problem["metrics"] = lambda x: {
        "y": branin(x["x1"], x["x2"]),
        "g": ((x["x1"] - 2.5) ** 2 + (x["x2"] - 7.5) ** 2 - 50.0) / 50.0,
    }
    # Branin's minimum at (pi, 2.275) is feasible.
    return problem


def _hartmann6_int():
    """Hartmann-6 with the last three coordinates on an 11-point integer grid."""
    parameters = [continuous(f"x{i}", 0.0, 1.0) for i in range(3)]
    parameters += [{"name": f"n{i}", "type": "integer", "lower": 0, "upper": 10} for i in range(3, 6)]
    return {
        "parameters": parameters,
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {
            "y": hartmann6([x["x0"], x["x1"], x["x2"]] + [x[f"n{i}"] / 10 for i in range(3, 6)])
        },
        # Computed by grid enumeration of the integer part and multi-start local search.
        "optimum": HARTMANN6_INT_OPTIMUM,
        "scale": 0.38,
    }


BRANIN_CATEGORIES = {"a": (0.0, 0.5), "b": (2.0, 0.0), "c": (-2.0, 1.0)}  # x1 shift, value offset


def _branin_categorical():
    return {
        "parameters": [
            continuous("x1", -5.0, 10.0),
            continuous("x2", 0.0, 15.0),
            {"name": "k", "type": "categorical", "choices": ["a", "b", "c"]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {
            "y": branin(x["x1"] - BRANIN_CATEGORIES[x["k"]][0], x["x2"]) + BRANIN_CATEGORIES[x["k"]][1]
        },
        # Category b: Branin's minimum at x1 = pi + 2 lies inside the box.
        "optimum": 0.397887357729738,
        "scale": 62.0,
    }


def _rosenbrock8():
    return {
        "parameters": [continuous(f"x{i}", -5.0, 10.0) for i in range(8)],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {"y": rosenbrock([x[f"x{i}"] for i in range(8)])},
        "optimum": 0.0,
        "scale": 5.8e5,
    }


def _ackley20():
    return {
        "parameters": [continuous(f"x{i}", -5.0, 10.0) for i in range(20)],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "metrics": lambda x: {"y": ackley([x[f"x{i}"] for i in range(20)])},
        "optimum": 0.0,
        "scale": 0.95,
    }


def _branin_currin():
    return {
        "parameters": [continuous("u1", 0.0, 1.0), continuous("u2", 0.0, 1.0)],
        "objectives": [
            {"name": "branin", "direction": "minimize"},
            {"name": "currin", "direction": "minimize"},
        ],
        "metrics": lambda x: {
            "branin": branin(15 * x["u1"] - 5, 15 * x["u2"]),
            "currin": currin(x["u1"], x["u2"]),
        },
        "reference_point": [18.0, 6.0],
        # Hypervolume of the true front against the reference point, as in BoTorch.
        "optimum": 59.36011874867746,
        "scale": 52.0,
        # Per-objective noise scales (added after mo-pilot ran with 52 for both).
        "scales": {"branin": 52.0, "currin": 2.6},
    }


def augmented_branin(x1, x2, s):
    """BoTorch's AugmentedBranin: Branin at s = 1, a correlated lower fidelity below."""
    t1 = x2 - (5.1 / (4 * math.pi**2) - 0.1 * (1 - s)) * x1**2 + 5 / math.pi * x1 - 6
    return t1**2 + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1) + 10


def _fidelity_problem(name, misleading):
    def value(x):
        s = x.get("s", 1.0)
        target = augmented_branin(x["x1"], x["x2"], s)
        if not misleading:
            return target
        # The low fidelity mirrors x1, so its minima lie elsewhere.
        return s * augmented_branin(x["x1"], x["x2"], 1.0) + (1 - s) * branin(5.0 - x["x1"], x["x2"])

    return {
        "parameters": [continuous("x1", -5.0, 10.0), continuous("x2", 0.0, 15.0), continuous("s", 0.0, 1.0)],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "fidelity": {"parameter": "s", "target": 1.0, "cost_intercept": 0.25, "cost_weight": 0.75},
        "metrics": lambda x: {"y": value(x)},
        "optimum": 0.397887357729738,
        "scale": 52.0,
    }


HARTMANN6_INT_OPTIMUM = -3.253105230874208

PROBLEMS = {
    "branin": _branin_problem,
    "hartmann6": _hartmann6_problem,
    "branin_constrained": _branin_constrained,
    "hartmann6_int": _hartmann6_int,
    "branin_cat": _branin_categorical,
    "rosenbrock8": _rosenbrock8,
    "ackley20": _ackley20,
    "branin_currin": _branin_currin,
    "branin_mf": lambda: _fidelity_problem("branin_mf", False),
    "branin_mf_misleading": lambda: _fidelity_problem("branin_mf_misleading", True),
}


def get(name):
    return PROBLEMS[name]()


def hypervolume_2d(points, reference):
    """Dominated hypervolume of minimization points against a reference point."""
    front = sorted({(a, b) for a, b in points if a < reference[0] and b < reference[1]})
    volume, best_b = 0.0, reference[1]
    for a, b in front:
        if b < best_b:
            volume += (reference[0] - a) * (best_b - b)
            best_b = b
    return volume

"""Tests for the improvements of 17 November 2026 (docs/plans)."""

import copy
import math
import random

import pytest

from bo_craft import propose, recommend
from bo_craft.specs import Error, canonical

LIGHT = {"num_restarts": 2, "raw_samples": 32, "mc_samples": 32, "fit_maxiter": 20, "acq_maxiter": 20}


def request(parameters, observations, config, objectives=None, constraints=None, q=1, pending=None):
    problem = {
        "schema_version": "bo.problem/0.1",
        "parameters": parameters,
        "objectives": objectives or [{"name": "y", "direction": "minimize"}],
    }
    if constraints:
        problem["constraints"] = constraints
    value = {
        "schema_version": "bo.request/0.1",
        "problem": problem,
        "config": config,
        "seed": 3,
        "q": q,
        "observations": observations,
    }
    if pending:
        value["pending"] = pending
    return value


def observation(i, x, values, variances=None):
    o = {"id": str(i), "trial_id": str(i), "x": x, "status": "completed", "values": values, "seed": i}
    if variances:
        o["variances"] = variances
    return o


MIXED = [
    {"name": "x", "type": "continuous", "lower": 0, "upper": 1},
    {"name": "n", "type": "integer", "lower": 1, "upper": 12},
    {"name": "k", "type": "categorical", "choices": ["a", "b", "c"]},
]


def mixed_data(count=10):
    rng = random.Random(1)
    data = []
    for i in range(count):
        x = {"x": rng.random(), "n": rng.randint(1, 12), "k": rng.choice("abc")}
        y = (x["x"] - 0.3) ** 2 + ((x["n"] - 7) / 10) ** 2 + {"a": 0.5, "b": 0.0, "c": 0.3}[x["k"]]
        data.append(observation(i, x, {"y": y + rng.gauss(0, 0.01)}))
    return data


def line(values, f):
    return [observation(i, {"x": v}, f(v)) for i, v in enumerate(values)]


X = [{"name": "x", "type": "continuous", "lower": 0, "upper": 1}]


def test_mixed_botorch_proposals_keep_declared_types():
    r = request(MIXED, mixed_data(), {"backend": "botorch", "initial_points": 8, **LIGHT}, q=2)
    result = propose(r)
    assert result["diagnostics"]["space"] == "mixed"
    for point in result["candidates"]:
        assert isinstance(point["n"], int) and 1 <= point["n"] <= 12
        assert point["k"] in ("a", "b", "c")
    assert propose(copy.deepcopy(r))["candidates"] == result["candidates"]


def test_optuna_gp_route_and_its_rejections():
    r = request(MIXED, mixed_data(), {"backend": "optuna", "sampler": "gp", "initial_points": 8})
    result = propose(r)
    assert result["diagnostics"]["sampler"] == "GPSampler"
    assert isinstance(result["candidates"][0]["n"], int)
    for config in [
        {"backend": "optuna", "sampler": "gp", "multivariate": True},
        {"backend": "optuna", "sampler": "gp", "trust_region": True},
        {"backend": "botorch", "sampler": "gp"},
    ]:
        with pytest.raises(Error) as caught:
            propose(request(MIXED, mixed_data(), config))
        assert caught.value.code == "unsupported_combination"


def test_model_recommendation_avoids_a_lucky_noisy_observation():
    # Three readings at x = 0.1 (true value 0.16), one of them lucky; the minimum is at 0.5.
    values = [0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    data = line(values, lambda v: {"y": (v - 0.5) ** 2})
    data[1]["values"]["y"], data[2]["values"]["y"], data[3]["values"]["y"] = -0.05, 0.3, 0.25
    r = request(X, data, {"backend": "botorch", "noise": "inferred"})
    assert recommend(r)["observations"][0]["x"] == {"x": 0.1}
    model = recommend(r, kind="model")
    assert model["kind"] == "model" and abs(model["points"][0]["x"]["x"] - 0.5) <= 0.1
    assert model["evidence"].startswith("observed points ranked by a fitted model")


def test_model_recommendation_respects_feasibility_and_fronts():
    values = [0.05, 0.2, 0.3, 0.4, 0.55, 0.7, 0.9]
    data = line(values, lambda v: {"a": (v - 0.2) ** 2, "b": (v - 0.8) ** 2, "g": v - 0.6})
    two = [{"name": "a", "direction": "minimize"}, {"name": "b", "direction": "minimize"}]
    r = request(
        X, data, {"backend": "botorch", "noise": "deterministic"}, objectives=two, constraints=[{"name": "g"}]
    )
    front = sorted(p["x"]["x"] for p in recommend(r, kind="model")["points"])
    assert front == [0.2, 0.3, 0.4, 0.55]  # 0.05 is dominated, 0.7 and 0.9 are infeasible
    for o in r["observations"]:
        o["values"]["g"] = 1.0
    with pytest.raises(Error) as caught:
        recommend(r, kind="model")
    assert caught.value.code == "no_feasible_recommendation"


def test_training_selection_keeps_the_incumbent():
    from bo_craft.backends.botorch import select_training

    problem = request(X, [], {"backend": "botorch"})["problem"]
    data = line([i / 100 for i in range(100)], lambda v: {"y": abs(v - 0.03)})
    selected, rule = select_training(problem, data, 20)
    ids = [o["id"] for o in selected]
    assert len(selected) == 20 and "3" in ids and ids[-1] == "99"
    assert rule == "best 5 plus latest 15"


def test_leave_one_out_matches_refits_with_fixed_hyperparameters():
    import numpy as np
    import torch

    from bo_craft.backends.botorch import fit_model

    data = line([0.05, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95], lambda v: {"y": math.sin(6 * v)})
    r = request(X, data, {"backend": "botorch", "noise": "inferred", **LIGHT})
    fit = fit_model(r["problem"], data, r["config"], 3)
    model = fit.model
    with torch.no_grad():
        x = model.train_inputs[0]
        y = model.train_targets.numpy()
        k = model.covar_module(x).to_dense().numpy() + float(model.likelihood.noise) * np.eye(len(y))
        m = model.mean_module(x).numpy()
    residuals = []
    for i in range(len(y)):
        keep = [j for j in range(len(y)) if j != i]
        weights = np.linalg.solve(k[np.ix_(keep, keep)], k[keep, i])
        mean = m[i] + weights @ (y[keep] - m[keep])
        residuals.append(y[i] - mean)
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    assert abs(fit.diagnostics["loo_rmse"][0] - rmse) <= 1e-3 * max(rmse, 1e-6)


def test_acquisition_iterations_are_separate_from_fitting(monkeypatch):
    import botorch.optim

    seen = {}
    original = botorch.optim.optimize_acqf

    def spy(*args, **kwargs):
        seen.update(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(botorch.optim, "optimize_acqf", spy)
    data = line([0.1, 0.3, 0.5, 0.7, 0.9], lambda v: {"y": v * v})
    config = {"backend": "botorch", "initial_points": 4, **LIGHT, "fit_maxiter": 3, "acq_maxiter": 17}
    propose(request(X, data, config))
    assert seen["maxiter"] == 17


def test_replication_is_declared_and_checked():
    rng = random.Random(4)
    values = [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9, 0.52, 0.48]
    data = line(values, lambda v: {"y": (v - 0.5) ** 2 + rng.gauss(0, 0.2)})
    config = {"backend": "botorch", "replication": "incumbent", "replication_threshold": 0.05, **LIGHT}
    result = propose(request(X, data, config, q=2))
    source = result["replicate_of"][0]
    assert source is not None and result["replicate_of"][1] is None
    assert canonical(result["candidates"][0]) == canonical(next(o["x"] for o in data if o["id"] == source))
    pending = [{"id": "p", "x": result["candidates"][0]}]
    again = propose(request(X, data, config, pending=pending))
    assert "replicate_of" not in again and not again["diagnostics"]["replication"]["replicated"]
    with pytest.raises(Error) as caught:
        propose(request(X, data, {**config, "noise": "deterministic"}))
    assert caught.value.code == "unsupported_combination"


def test_undeclared_repeats_are_still_rejected(monkeypatch):
    import bo_craft.backends.botorch as backend

    data = line([0.1, 0.3, 0.5, 0.7, 0.9], lambda v: {"y": v})
    monkeypatch.setattr(backend, "propose", lambda request: ([{"x": 0.3}], {"replicate_of": ["1"]}))
    with pytest.raises(Error) as caught:
        propose(request(X, data, {"backend": "botorch"}))
    assert caught.value.code == "backend_protocol"  # replication is not enabled
    monkeypatch.setattr(backend, "propose", lambda request: ([{"x": 0.3}], {}))
    with pytest.raises(Error) as caught:
        propose(request(X, data, {"backend": "botorch"}))
    assert caught.value.code == "duplicate_candidate"


def test_trust_region_side_length_is_replayed_from_the_ledger():
    from bo_craft.backends.botorch import trust_region_state

    problem = request(X, [], {"backend": "botorch"})["problem"]
    improving = line([0.9 - 0.05 * i for i in range(11)], lambda v: {"y": v})
    assert trust_region_state(problem, improving, 2, 2) == (1.6, 0)  # three successes doubled 0.8 once
    flat = line([0.5] * 2 + [0.9] * 8, lambda v: {"y": v})
    assert trust_region_state(problem, flat, 2, 2) == (0.2, 0)  # two failure streaks of four


def test_outcome_transforms_are_checked_and_back_transformed():
    data = line([0.05, 0.2, 0.4, 0.55, 0.7, 0.9], lambda v: {"y": math.exp(5 * v)})
    config = {"backend": "botorch", "noise": "deterministic", "outcome_transform": "log", **LIGHT}
    result = recommend(request(X, data, config), kind="posterior")
    assert result["outcome_transform"] == "log" and "mean" not in result
    assert result["median"] > 0 and result["pointwise_model_interval"][0] > 0
    data[0]["values"]["y"] = -1.0
    with pytest.raises(Error) as caught:
        recommend(request(X, data, config), kind="model")
    assert caught.value.code == "invalid_observation"


def test_wrapper_matches_direct_botorch():
    """Same model, seed and acquisition settings give the same candidate as direct BoTorch."""
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood

    params = [{"name": f"x{i}", "type": "continuous", "lower": -1, "upper": 1} for i in range(2)]
    rng = random.Random(7)
    data = []
    for i in range(9):
        x = {"x0": rng.uniform(-1, 1), "x1": rng.uniform(-1, 1)}
        data.append(observation(i, x, {"y": x["x0"] ** 2 + 0.5 * x["x1"] + rng.gauss(0, 0.05)}))
    config = {"backend": "botorch", "initial_points": 8, **LIGHT}
    ours = propose(request(params, data, config))["candidates"][0]

    torch.set_num_threads(1)
    torch.manual_seed(3)
    train_x = torch.tensor([[(o["x"][k] + 1) / 2 for k in ("x0", "x1")] for o in data], dtype=torch.double)
    train_y = torch.tensor([[-o["values"]["y"]] for o in data], dtype=torch.double)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(
        ExactMarginalLogLikelihood(model.likelihood, model),
        optimizer_kwargs={"options": {"maxiter": 20}},
        max_attempts=1,
    )
    acq = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(torch.Size([32]), seed=3),
        objective=GenericMCObjective(lambda z, X=None: z[..., 0]),
        prune_baseline=True,
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    direct, _ = optimize_acqf(
        acq, bounds, 1, num_restarts=2, raw_samples=32, options={"batch_limit": 2, "maxiter": 20}
    )
    expected = [2 * float(v) - 1 for v in direct[0]]
    assert abs(ours["x0"] - expected[0]) < 1e-6 and abs(ours["x1"] - expected[1]) < 1e-6


def test_knowledge_gradient_fantasies_are_configurable():
    from bo_craft.backends.botorch import Space, fit_model, knowledge_gradient

    params = [
        {"name": "x", "type": "continuous", "lower": 0, "upper": 1},
        {"name": "s", "type": "continuous", "lower": 0, "upper": 1},
    ]
    data = [observation(i, {"x": i / 7, "s": (i % 2) * 1.0}, {"y": (i / 7 - 0.4) ** 2}) for i in range(8)]
    r = request(params, data, {"backend": "botorch", "num_fantasies": 4, **LIGHT})
    r["problem"]["fidelity"] = {"parameter": "s", "target": 1.0, "cost_intercept": 1.0, "cost_weight": 4.0}
    fit = fit_model(r["problem"], data, r["config"], 3)
    acq = knowledge_gradient(fit.model, r["problem"], r["config"], Space(r["problem"]))
    assert acq.num_fantasies == 4

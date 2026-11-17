import copy

import pytest

from bo_craft import propose, recommend
from bo_craft.specs import Error, aggregate, validate_request


@pytest.fixture
def bo_request():
    return {
        "schema_version": "bo.request/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": -2, "upper": 2}],
            "objectives": [{"name": "score", "direction": "minimize"}],
        },
        "config": {
            "backend": "botorch",
            "initial_points": 4,
            "noise": "known",
            "fit_maxiter": 8,
            "num_restarts": 2,
            "raw_samples": 16,
            "mc_samples": 16,
        },
        "seed": 41,
        "q": 1,
        "observations": [],
        "pending": [],
    }


def populate(r):
    r["observations"] = [
        {
            "id": str(i),
            "trial_id": str(i),
            "x": {"x": x},
            "status": "completed",
            "values": {"score": x * x},
            "variances": {"score": 0.01},
            "seed": i,
        }
        for i, x in enumerate([-1.8, -0.8, 0.5, 1.4])
    ]


def test_noisy_model(bo_request):
    populate(bo_request)
    result = propose(bo_request)
    assert -2 <= result["candidates"][0]["x"] <= 2
    assert result["diagnostics"]["acquisition"] == "log_nei"
    again = propose(bo_request)
    assert again["candidates"] == result["candidates"]


def test_posterior_prediction_is_not_evaluated_evidence(bo_request):
    populate(bo_request)
    result = recommend(bo_request, kind="posterior")
    assert -2 <= result["x"]["x"] <= 2
    assert result["latent_standard_deviation"] >= 0
    assert result["evidence"].startswith("predicted")


def test_posterior_matches_independent_linear_solve(bo_request):
    import numpy as np
    import torch

    from bo_craft.backends.botorch import model_for

    populate(bo_request)
    model, x, _, _ = model_for(bo_request)
    point = torch.tensor([[0.55]], dtype=torch.double)
    with torch.no_grad():
        k = model.covar_module(x, x).to_dense().numpy()
        noise = model.likelihood.noise.detach().numpy().reshape(-1)
        targets = model.train_targets.detach().numpy()
        mean = model.mean_module(x).detach().numpy()
        cross = model.covar_module(point, x).to_dense().numpy().reshape(-1)
        estimate = float(model.mean_module(point).item()) + cross @ np.linalg.solve(
            k + np.diag(noise), targets - mean
        )
        scale = float(model.outcome_transform.stdvs.item())
        shift = float(model.outcome_transform.means.item())
        assert abs(float(model.posterior(point).mean.item()) - (estimate * scale + shift)) < 1e-8


def test_constraints_and_two_objectives(bo_request):
    populate(bo_request)
    bo_request["problem"]["objectives"].append({"name": "other", "direction": "maximize"})
    bo_request["problem"]["constraints"] = [{"name": "g"}]
    bo_request["config"]["reference_point"] = [8, -8]
    for o in bo_request["observations"]:
        x = o["x"]["x"]
        o["values"].update(other=-((x - 1) ** 2), g=x - 1)
        o["variances"].update(other=0.01, g=0.01)
    assert propose(bo_request)["diagnostics"]["acquisition"] == "log_nehvi"
    assert recommend(bo_request)["kind"] == "empirical"


def test_pending_and_failure(bo_request):
    populate(bo_request)
    bo_request["observations"].append({"id": "bad", "trial_id": "bad", "x": {"x": 1.9}, "status": "failed"})
    bo_request["pending"] = [{"id": "future", "x": {"x": 0.0}}]
    result = propose(bo_request)
    assert result["candidates"] != [{"x": 0.0}]
    assert "bad" not in result["diagnostics"]["training_ids"]


@pytest.mark.parametrize("change", ["unknown", "negative_variance", "duplicate", "nonfinite"])
def test_rejections(bo_request, change):
    populate(bo_request)
    if change == "unknown":
        bo_request["extra"] = True
    elif change == "negative_variance":
        bo_request["observations"][0]["variances"]["score"] = -1
    elif change == "duplicate":
        bo_request["observations"].append(copy.deepcopy(bo_request["observations"][0]))
    else:
        bo_request["observations"][0]["values"]["score"] = float("nan")
    with pytest.raises(Error):
        validate_request(bo_request)


def test_mean_variance(bo_request):
    populate(bo_request)
    replicates = copy.deepcopy(bo_request["observations"][:2])
    for r in replicates:
        r["x"] = {"x": 0.0}
    replicates[0]["values"]["score"] = 1
    replicates[1]["values"]["score"] = 3
    result = aggregate(replicates, "mean")
    assert result["values"]["score"] == 2
    assert result["variances"]["score"] == 1
    replicates[1]["seed"] = replicates[0]["seed"]
    with pytest.raises(Error):
        aggregate(replicates, "bad")


def test_conditional_tpe(bo_request):
    bo_request["config"] = {"backend": "optuna", "initial_points": 2}
    bo_request["problem"]["parameters"] = [
        {"name": "kind", "type": "categorical", "choices": ["a", "b"]},
        {"name": "n", "type": "integer", "lower": 1, "upper": 4, "active_if": {"kind": ["a"]}},
        {
            "name": "x",
            "type": "continuous",
            "lower": 0.1,
            "upper": 1,
            "log": True,
            "active_if": {"kind": ["b"]},
        },
    ]
    bo_request["q"] = 4
    result = propose(bo_request)
    assert len(result["candidates"]) == 4
    assert all(("n" in p) != ("x" in p) for p in result["candidates"])

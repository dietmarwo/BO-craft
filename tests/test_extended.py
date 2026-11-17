import json

import pytest

from bo_craft import propose
from bo_craft.pruning import should_prune
from bo_craft.specs import Error, validate_request


def base():
    return {
        "schema_version": "bo.request/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
            "objectives": [{"name": "score", "direction": "minimize"}],
        },
        "seed": 41,
        "q": 1,
        "config": {
            "backend": "botorch",
            "initial_points": 2,
            "noise": "known",
            "fit_maxiter": 4,
            "num_restarts": 1,
            "raw_samples": 16,
            "mc_samples": 16,
        },
        "observations": [],
    }


def test_multifidelity():
    r = base()
    r["problem"]["parameters"].append({"name": "fidelity", "type": "continuous", "lower": 0.0, "upper": 1.0})
    r["problem"]["fidelity"] = {
        "parameter": "fidelity",
        "target": 1.0,
        "cost_intercept": 0.1,
        "cost_weight": 1.0,
    }
    r["config"]["acquisition"] = "multifidelity_kg"
    for i, (x, f) in enumerate([(0.1, 0.2), (0.8, 1.0), (0.4, 0.7), (0.6, 0.3)]):
        r["observations"].append(
            {
                "id": str(i),
                "trial_id": str(i),
                "x": {"x": x, "fidelity": f},
                "status": "completed",
                "values": {"score": (x - 0.3) ** 2 + (1 - f) * 0.1},
                "variances": {"score": 0.001},
            }
        )
    result = propose(r)
    assert result["diagnostics"]["acquisition"] == "multifidelity_kg"
    assert 0 <= result["candidates"][0]["fidelity"] <= 1


def test_known_constraint():
    r = base()
    r["problem"]["input_constraints"] = [{"coefficients": {"x": 1}, "upper": 0.4}]
    for i, x in enumerate([0.05, 0.3]):
        r["observations"].append(
            {
                "id": str(i),
                "trial_id": str(i),
                "x": {"x": x},
                "status": "completed",
                "values": {"score": x * x},
                "variances": {"score": 0.001},
            }
        )
    assert propose(r)["candidates"][0]["x"] <= 0.4 + 1e-8


def test_pruning_retains_execution_uncertainty():
    result = should_prune([{1: 0.2, 2: 0.1} for _ in range(6)], {1: 5.0, 2: 4.0}, startup_trials=2)
    assert result["should_prune"]
    assert not result["execution_stopped"]


def test_unsupported_before_initial_design():
    r = base()
    r["problem"]["parameters"] = [
        {"name": "kind", "type": "categorical", "choices": ["a", "b"]},
        {"name": "x", "type": "continuous", "lower": 0, "upper": 1, "active_if": {"kind": ["a"]}},
    ]
    with pytest.raises(Error, match="Optuna"):
        propose(r)


def test_invalid_conditional_cycle():
    r = base()
    r["problem"]["parameters"] = [
        {"name": "a", "type": "categorical", "choices": ["x"], "active_if": {"a": ["x"]}}
    ]
    with pytest.raises(Error):
        validate_request(r)


def test_read_only_bo_mcp_query(monkeypatch):
    from bo_craft.backends import services

    r = base()
    r["config"] = {"backend": "bo-mcp", "service": {"endpoint": "http://localhost:1234", "study": "known-id"}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return json.dumps({"suggestions": [{"id": "s1", "parameters": {"x": 0.2}}]}).encode()

    calls = []

    def open_request(req, **kwargs):
        calls.append(req.full_url)
        return Response()

    monkeypatch.setattr(services.urllib.request, "urlopen", open_request)
    a = propose(r)
    b = propose(r)
    assert a["candidates"] == b["candidates"] == [{"x": 0.2}]
    assert all(url.endswith("/query") for url in calls)


def test_explicit_ref_point_required():
    r = base()
    r["problem"]["objectives"].append({"name": "y", "direction": "maximize"})
    with pytest.raises(Error, match="reference"):
        propose(r)

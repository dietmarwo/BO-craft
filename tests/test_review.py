"""Regression tests for the review fixes of the first source release."""

import json

import pytest

from bo_craft import cli, propose
from bo_craft.backends.services import vizier_point
from bo_craft.pruning import should_prune
from bo_craft.specs import Error


def request(backend="botorch"):
    return {
        "schema_version": "bo.request/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
            "objectives": [{"name": "score", "direction": "minimize"}],
        },
        "seed": 3,
        "q": 1,
        "config": {"backend": backend, "initial_points": 2, "noise": "inferred", "fit_maxiter": 4},
        "observations": [
            {
                "id": str(i),
                "trial_id": str(i),
                "x": {"x": x},
                "status": "completed",
                "values": {"score": (x - 0.3) ** 2},
            }
            for i, x in enumerate([0.1, 0.5, 0.9])
        ],
        "pending": [],
    }


def test_a_failed_gp_fit_is_a_typed_error(monkeypatch):
    pytest.importorskip("botorch")
    import botorch.fit
    from botorch.exceptions.errors import ModelFittingError

    def fail(*args, **kwargs):
        raise ModelFittingError("injected")

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", fail)
    with pytest.raises(Error) as caught:
        propose(request())
    assert caught.value.code == "model_fit_failed"


def test_cli_reports_file_and_dependency_errors_as_json(tmp_path, capsys):
    assert cli.main(["validate", str(tmp_path / "missing.json")]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "io_error"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request()))
    assert cli.main(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"valid": True}


def test_vizier_points_take_the_declared_types():
    problem = {
        "parameters": [
            {"name": "n", "type": "integer", "lower": 1, "upper": 9},
            {"name": "kind", "type": "categorical", "choices": ["a", "b"]},
        ]
    }
    point = vizier_point(problem, {"n": 3.0, "kind": "b"})
    assert point == {"n": 3, "kind": "b"} and type(point["n"]) is int
    with pytest.raises(Error) as caught:
        vizier_point(problem, {"n": 3.5, "kind": "a"})
    assert caught.value.code == "backend_protocol"


def test_pruning_rejects_completed_curves_with_non_integer_steps():
    with pytest.raises(Error):
        should_prune([{"1": 0.2, "2": 0.1}], {1: 5.0})

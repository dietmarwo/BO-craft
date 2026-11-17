"""Optional real local Vizier service logic; remote transport is a separate gate."""

import pytest


def test_real_vizier_outstanding_suggestions_are_reconciled():
    pytest.importorskip("vizier")
    from bo_craft import propose

    request = {
        "schema_version": "bo.request/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
            "objectives": [{"name": "score", "direction": "minimize"}],
        },
        "seed": 41,
        "q": 1,
        "config": {
            "backend": "vizier",
            "service": {"endpoint": "NO_ENDPOINT", "study": "reconcile-test", "backend": "RANDOM_SEARCH"},
        },
        "observations": [
            {"id": "one", "trial_id": "one", "x": {"x": 0.4}, "values": {"score": 0.2}, "status": "completed"}
        ],
    }
    first = propose(request)
    second = propose(request)
    assert first["candidates"] == second["candidates"]
    assert first["diagnostics"]["trial_ids"] == second["diagnostics"]["trial_ids"]

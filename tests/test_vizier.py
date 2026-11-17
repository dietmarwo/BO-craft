"""Real Vizier service logic (local in-process service and a local gRPC server).

Skipped without the optional `vizier` extra; CI runs them in a separate job.
"""

import pytest

vizier = pytest.importorskip("vizier")


def request(study, observations, pending=None, endpoint="NO_ENDPOINT", q=1, seed=41):
    value = {
        "schema_version": "bo.request/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": 0.0, "upper": 1.0}],
            "objectives": [{"name": "score", "direction": "minimize"}],
        },
        "seed": seed,
        "q": q,
        "config": {
            "backend": "vizier",
            "service": {"endpoint": endpoint, "study": study, "backend": "RANDOM_SEARCH"},
        },
        "observations": observations,
    }
    if pending:
        value["pending"] = pending
    return value


def observation(i, x, status="completed"):
    o = {"id": f"o{i}", "trial_id": f"t{i}", "x": {"x": x}, "status": status}
    if status == "completed":
        o["values"] = {"score": (x - 0.3) ** 2}
    return o


@pytest.fixture(scope="module")
def server():
    from vizier.service import servers

    # The default database is a file inside the installed package; tests use memory.
    return servers.DefaultVizierServer(database_url="sqlite:///:memory:")


def trials(endpoint, study_id):
    from vizier.service import clients

    clients.environment_variables.server_endpoint = endpoint
    return list(clients.Study.from_owner_and_id("bo-craft", study_id).trials().get())


def test_real_vizier_outstanding_suggestions_are_reconciled():
    from bo_craft import propose

    value = request("reconcile-test", [observation(0, 0.4)])
    first = propose(value)
    second = propose(value)
    assert first["candidates"] == second["candidates"]
    assert first["diagnostics"]["trial_ids"] == second["diagnostics"]["trial_ids"]


def test_a_campaign_study_imports_each_observation_once(server):
    """The plan's gate: 50 proposals, each observation imported exactly once."""
    from bo_craft import propose

    observations = [observation(0, 0.9), observation(1, 0.1)]
    study_id = None
    for i in range(2, 52):
        result = propose(request("fifty", observations, endpoint=server.endpoint, seed=i))
        counts = result["diagnostics"]["reconciliation"]
        study_id = result["diagnostics"]["study"]
        if i == 2:
            assert counts == {
                "imported": 2,
                "completed_suggestions": 0,
                "withdrawn": 0,
                "already_imported": 0,
            }
        else:
            # The previous candidate completes its own suggestion; nothing else moves.
            assert counts == {
                "imported": 0,
                "completed_suggestions": 1,
                "withdrawn": 0,
                "already_imported": len(observations) - 1,
            }
        observations.append(observation(i, result["candidates"][0]["x"]))
    stored = trials(server.endpoint, study_id)
    ids = [t.metadata.ns("bo-craft").get("observation_id") for t in stored]
    marked = [i for i in ids if i]
    assert len(marked) == len(set(marked)) == 51  # o0 .. o50; o51 is not observed yet
    assert sum(t.status.name == "COMPLETED" for t in stored) == 51
    assert sum(t.status.name == "ACTIVE" for t in stored) == 1


def test_failed_work_and_abandoned_suggestions_are_withdrawn(server):
    from bo_craft import propose

    base = [observation(0, 0.5), observation(1, 0.2)]
    first = propose(request("withdraw", base, endpoint=server.endpoint, q=3))
    a, b, c = (point["x"] for point in first["candidates"])
    # a failed, b is still being evaluated, c was abandoned by the caller.
    later = request(
        "withdraw",
        [*base, observation(2, a, status="failed")],
        pending=[{"id": "p", "x": {"x": b}}],
        endpoint=server.endpoint,
        seed=7,
    )
    result = propose(later)
    assert result["diagnostics"]["reconciliation"]["withdrawn"] == 2
    stored = trials(server.endpoint, result["diagnostics"]["study"])
    open_points = sorted(t.parameters["x"].value for t in stored if t.status.name in ("ACTIVE", "REQUESTED"))
    assert b in open_points and a not in open_points and c not in open_points
    assert result["candidates"][0]["x"] != b


def test_a_changed_observation_is_a_diverged_study(server):
    from bo_craft import propose
    from bo_craft.specs import Error

    observations = [observation(0, 0.5), observation(1, 0.2)]
    propose(request("diverge", observations, endpoint=server.endpoint))
    observations[0]["values"]["score"] = 99.0
    with pytest.raises(Error) as caught:
        propose(request("diverge", observations, endpoint=server.endpoint, seed=8))
    assert caught.value.code == "study_diverged"


def test_integer_and_categorical_suggestions_keep_their_types(server):
    from bo_craft import propose

    # A wide integer range: random search may repeat points of a small discrete
    # space within one batch, which BO-craft refuses as a duplicate.
    value = request("types", [], endpoint=server.endpoint, q=2)
    value["problem"]["parameters"] = [
        {"name": "n", "type": "integer", "lower": 1, "upper": 1000000},
        {"name": "k", "type": "categorical", "choices": ["a", "b"]},
    ]
    for point in propose(value)["candidates"]:
        assert isinstance(point["n"], int) and point["k"] in ("a", "b")

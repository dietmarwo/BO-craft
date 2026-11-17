"""Optional external-study adapters with conservative reconciliation semantics."""

import json
import urllib.error
import urllib.request

from ..specs import Error, canonical, digest, require


def propose(request):
    service = request["config"].get("service")
    require(service is not None, "invalid_spec", "external backend requires service configuration")
    if request["config"]["backend"] == "vizier":
        return vizier(request, service)
    return bo_mcp(request, service)


def vizier(request, service):
    try:
        from vizier.service import clients
    except ImportError as exc:
        raise Error("missing_dependency", "install bo-craft[vizier]") from exc
    p = request["problem"]
    require(
        not p.get("constraints") and not p.get("fidelity") and not p.get("input_constraints"),
        "unsupported_combination",
        "Vizier adapter currently flat unconstrained spaces",
    )
    require(
        not any(v.get("active_if") for v in p["parameters"]),
        "unsupported_combination",
        "conditional Vizier designer support is not validated",
    )
    # This changes Vizier's process-global environment: use an isolated proposal
    # process, as experiment-craft does; no concurrent endpoints in one process.
    clients.environment_variables.server_endpoint = service["endpoint"]
    from vizier.service import pyvizier as vz

    config = vz.StudyConfig(algorithm=service.get("backend", "RANDOM_SEARCH"))
    for v in p["parameters"]:
        if v["type"] == "categorical":
            config.search_space.root.add_categorical_param(v["name"], v["choices"])
        elif v["type"] == "integer":
            config.search_space.root.add_int_param(v["name"], v["lower"], v["upper"])
        else:
            config.search_space.root.add_float_param(
                v["name"],
                v["lower"],
                v["upper"],
                scale_type=vz.ScaleType.LOG if v.get("log") else vz.ScaleType.LINEAR,
            )
    for metric in p["objectives"]:
        goal = (
            vz.ObjectiveMetricGoal.MINIMIZE
            if metric["direction"] == "minimize"
            else vz.ObjectiveMetricGoal.MAXIMIZE
        )
        config.metric_information.append(vz.MetricInformation(metric["name"], goal=goal))
    # A request-scoped deterministic study makes observation ingestion rebuildable
    # without mutating a long-lived external study with uncertain result updates.
    study_id = service["study"] + "-" + digest(request)[:24]
    study = clients.Study.from_study_config(config, owner="bo-craft", study_id=study_id)
    existing = list(study.trials().get())
    imported = {
        t.metadata.ns("bo-craft").get("observation_id"): t
        for t in existing
        if t.metadata.ns("bo-craft").get("observation_id")
    }
    for o in request.get("observations", []):
        if o["status"] != "completed":
            continue
        if o["id"] in imported:
            require(
                imported[o["id"]].metadata.ns("bo-craft").get("sha256") == digest(o),
                "study_diverged",
                "Vizier observation differs from request ledger",
            )
            continue
        trial = vz.Trial(parameters=o["x"])
        trial.metadata.ns("bo-craft")["observation_id"] = o["id"]
        trial.metadata.ns("bo-craft")["sha256"] = digest(o)
        trial.complete(vz.Measurement(o["values"]))
        study.add_trial(trial)
    # Vizier reuses outstanding trials for a client_id, making lost-ack reads
    # reconcilable. Never complete the returned trials inside a proposal call.
    trials = study.suggest(count=request["q"], client_id=digest(request))
    points = [{k: getattr(v, "value", v) for k, v in t.parameters.items()} for t in trials]
    return points, {
        "study": study_id,
        "trial_ids": [t.id for t in trials],
        "algorithm": service.get("backend", "RANDOM_SEARCH"),
        "scope": "request-scoped study projection; outstanding suggestions retained",
    }


def bo_mcp(request, service):
    require(
        not request.get("observations"),
        "unsupported_combination",
        "BO-MCP bridge reads pre-generated suggestions; result synchronization remains external",
    )
    # Generation has no idempotency key in the inspected API. Read existing
    # suggestions only, rather than repeat a mutation after a lost response.
    url = service["endpoint"].rstrip("/") + "/api/v1/suggestions/" + service["study"] + "/query"
    data = canonical({"status_filter": "pending", "limit": 500, "verbosity": "standard"})
    call = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(call, timeout=service.get("timeout_seconds", 30)) as response:
            raw = response.read(4 * 1024**2 + 1)
        require(len(raw) <= 4 * 1024**2, "size_limit", "external response exceeds limit")
        value = json.loads(raw)
    except (urllib.error.URLError, ValueError) as exc:
        raise Error("service_unavailable", str(exc)) from exc
    rows = value.get("suggestions", [])
    pending = {canonical(o["x"]) for o in request.get("pending", [])}
    rows = [r for r in rows if canonical(r.get("parameters", {})) not in pending][: request["q"]]
    require(
        len(rows) == request["q"],
        "suggestions_unavailable",
        "generate suggestions in the external service first",
    )
    return [r["parameters"] for r in rows], {
        "external_suggestion_ids": [str(r["id"]) for r in rows],
        "scope": "experimental read-only BO-MCP REST bridge; no automatic campaign mutation",
        "endpoint_protocol": "suggestions query; live BO-MCP deployment acceptance pending",
    }

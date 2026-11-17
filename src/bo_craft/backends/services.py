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


def vizier_point(problem, parameters):
    """A Vizier suggestion as a point of the problem's declared types.

    Vizier reports values as floats or strings; an integer parameter's value
    must be integral, and is returned as an int.
    """
    point = {}
    for v in problem["parameters"]:
        value = parameters[v["name"]]
        value = getattr(value, "value", value)
        if v["type"] == "integer":
            require(
                isinstance(value, (int, float)) and float(value).is_integer(),
                "backend_protocol",
                f"non-integral value for integer parameter {v['name']}",
            )
            value = int(value)
        point[v["name"]] = value
    return point


def vizier(request, service):
    """One study per campaign (study name, problem and algorithm).

    Observations are imported once, keyed by observation ID and digest. A
    completed observation at the point of an outstanding Vizier suggestion
    completes that suggestion, so Vizier's own trials never stay active; a
    failed one withdraws it. Suggestions that are neither pending nor held
    for this request are withdrawn. With an ephemeral service (NO_ENDPOINT,
    a fresh process) the study starts empty and the whole ledger is imported,
    which gives the same result.
    """
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
    try:
        from vizier.service import clients
    except ImportError as exc:
        raise Error("missing_dependency", "install bo-craft[vizier]") from exc
    # This changes Vizier's process-global environment: use an isolated proposal
    # process, as experiment-craft does; no concurrent endpoints in one process.
    clients.environment_variables.server_endpoint = service["endpoint"]
    from vizier.service import pyvizier as vz

    algorithm = service.get("backend", "RANDOM_SEARCH")
    config = vz.StudyConfig(algorithm=algorithm)
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
    study_id = service["study"] + "-" + digest({"problem": p, "algorithm": algorithm})[:24]
    study = clients.Study.from_study_config(config, owner="bo-craft", study_id=study_id)
    client_id = digest(request)

    def tag(trial):
        return trial.metadata.ns("bo-craft")

    existing = list(study.trials().get())
    imported = {tag(t).get("observation_id"): t for t in existing if tag(t).get("observation_id")}
    open_states = (vz.TrialStatus.ACTIVE, vz.TrialStatus.REQUESTED)
    outstanding = {}
    for t in existing:
        if t.status in open_states and not tag(t).get("observation_id"):
            outstanding.setdefault(canonical(vizier_point(p, t.parameters)), []).append(t)
    counts = {"imported": 0, "completed_suggestions": 0, "withdrawn": 0, "already_imported": 0}

    def mark(observation):
        metadata = vz.Metadata()
        metadata.ns("bo-craft")["observation_id"] = observation["id"]
        metadata.ns("bo-craft")["sha256"] = digest(observation)
        return metadata

    for o in request.get("observations", []):
        if o["id"] in imported:
            require(
                tag(imported[o["id"]]).get("sha256") == digest(o),
                "study_diverged",
                "Vizier observation differs from request ledger",
            )
            counts["already_imported"] += 1
            continue
        match = outstanding.get(canonical(o["x"]))
        suggestion = study.get_trial(match.pop(0).id) if match else None
        if o["status"] != "completed":
            # Failed, pruned, cancelled or unknown work is no measurement:
            # withdraw the suggestion it came from, import nothing.
            if suggestion is not None:
                suggestion.delete()
                counts["withdrawn"] += 1
            continue
        if suggestion is not None:
            suggestion.update_metadata(mark(o))
            suggestion.complete(vz.Measurement(o["values"]))
            counts["completed_suggestions"] += 1
        else:
            trial = vz.Trial(parameters=o["x"])
            # Metadata.update does not copy namespaces; set the tags directly.
            trial.metadata.ns("bo-craft")["observation_id"] = o["id"]
            trial.metadata.ns("bo-craft")["sha256"] = digest(o)
            trial.complete(vz.Measurement(o["values"]))
            study.add_trial(trial)
            counts["imported"] += 1
    pending = {canonical(o["x"]) for o in request.get("pending", [])}
    for key, trials in outstanding.items():
        for t in trials:
            # Pending evaluations and this request's own outstanding answer stay.
            if key in pending or t.assigned_worker == client_id:
                continue
            study.get_trial(t.id).delete()
            counts["withdrawn"] += 1
    # Vizier returns this client's outstanding trials first, so a repeated
    # request after a lost answer gets the same suggestions.
    trials = study.suggest(count=request["q"], client_id=client_id)
    points = [vizier_point(p, t.parameters) for t in trials]
    return points, {
        "study": study_id,
        "trial_ids": [t.id for t in trials],
        "algorithm": algorithm,
        "scope": "campaign study; observations imported once by ID and digest",
        "reconciliation": counts,
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

"""Backend dispatch; all suggestions are validated before artifact publication."""

import importlib.metadata
import math
import random
import time

from .specs import (
    Error,
    active,
    canonical,
    decode,
    digest,
    require,
    schema_check,
    validate_point,
    validate_request,
)


def capabilities():
    return {
        "schema_version": "bo.capabilities/0.1",
        "backends": {
            "random": {"spaces": ["continuous", "integer", "categorical", "conditional"], "model": False},
            "botorch": {
                "spaces": ["continuous", "integer", "categorical"],
                "noise": ["known", "inferred", "deterministic"],
                "constraints": True,
                "multiobjective": True,
                "multifidelity": "scalar continuous",
                "pending": True,
                "outcome_transforms": ["standardize", "log", "bilog"],
                "replication": ["none", "incumbent (experimental)"],
                "trust_region": "experimental, scalar unconstrained continuous",
                "device": "cpu",
                "precision": "float64",
            },
            "optuna": {
                "samplers": {
                    "tpe": {
                        "spaces": ["continuous", "integer", "categorical", "conditional"],
                        "noise_model": False,
                    },
                    "gp": {
                        "spaces": ["continuous", "integer", "categorical"],
                        "noise": ["inferred", "deterministic"],
                    },
                },
                "constraints": True,
                "multiobjective": True,
                "pending": True,
                "pruning": "single objective via intermediate contract",
            },
            "vizier": {
                "status": "optional service adapter",
                "requires": "existing study and capability contract",
            },
            "bo-mcp": {
                "status": "optional REST adapter",
                "requires": "existing campaign and capability contract",
            },
        },
        "recommendations": ["empirical", "model", "posterior"],
        "execution": "caller owned; adaptive campaigns in experiment-craft",
    }


def random_points(request, count=None):
    problem = request["problem"]
    rng = random.Random(request["seed"])
    occupied = {canonical(o["x"]) for o in request.get("observations", []) + request.get("pending", [])}
    points = []
    for _ in range(10000):
        x = {}
        for p in problem["parameters"]:
            if not active(p, x):
                continue
            if p["type"] == "categorical":
                x[p["name"]] = rng.choice(p["choices"])
            elif p["type"] == "integer":
                if p.get("log"):
                    value = round(math.exp(rng.uniform(math.log(p["lower"]), math.log(p["upper"]))))
                    x[p["name"]] = min(p["upper"], max(p["lower"], value))
                else:
                    x[p["name"]] = rng.randint(p["lower"], p["upper"])
            else:
                value = rng.random()
                x[p["name"]] = decode({"parameters": [p]}, [value])[p["name"]]
        try:
            validate_point(problem, x)
        except Error as exc:
            if exc.code == "input_infeasible":
                continue
            raise
        key = canonical(x)
        if key not in occupied:
            points.append(x)
            occupied.add(key)
        if len(points) == (count or request["q"]):
            return points
    raise Error("candidate_exhausted", "no distinct feasible candidate within 10000 bounded attempts")


def propose(request):
    validate_request(request)
    began = time.monotonic()
    name = request["config"]["backend"]
    if name == "random":
        points, diagnostic = random_points(request), {"model": "random"}
    elif name == "botorch":
        from .backends.botorch import propose as backend

        points, diagnostic = backend(request)
    elif name == "optuna":
        from .backends.optuna import propose as backend

        points, diagnostic = backend(request)
    else:
        from .backends.services import propose as backend

        points, diagnostic = backend(request)
    require(len(points) == request["q"], "backend_protocol", "wrong suggestion cardinality")
    replicate_of = diagnostic.pop("replicate_of", None) or [None] * len(points)
    require(len(replicate_of) == len(points), "backend_protocol", "replicate list cardinality")
    completed = {o["id"]: o for o in request.get("observations", []) if o["status"] == "completed"}
    observed = {canonical(o["x"]) for o in request.get("observations", [])}
    occupied = {canonical(o["x"]) for o in request.get("pending", [])}
    for point, source in zip(points, replicate_of):
        validate_point(request["problem"], point)
        key = canonical(point)
        if source is not None:
            # Only a declared replicate of a completed observation may repeat a point.
            require(
                request["config"].get("replication", "none") != "none"
                and source in completed
                and canonical(completed[source]["x"]) == key,
                "backend_protocol",
                "undeclared or mismatched replicate",
            )
        else:
            require(key not in observed, "duplicate_candidate", "backend suggested an existing point")
        require(key not in occupied, "duplicate_candidate", "backend suggested a pending point")
        occupied.add(key)
    diagnostic.update(
        {
            "backend": name,
            "proposal_seconds": time.monotonic() - began,
            "observation_ids": [o["id"] for o in request.get("observations", [])],
            "pending_ids": [o["id"] for o in request.get("pending", [])],
        }
    )
    versions = {}
    for package in ["bo-craft", "torch", "botorch", "gpytorch", "optuna"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    diagnostic["versions"] = versions
    result = {
        "schema_version": "bo.proposal/0.1",
        "request_sha256": digest(request),
        "problem_sha256": digest(request["problem"]),
        "candidates": points,
        "diagnostics": diagnostic,
    }
    if any(source is not None for source in replicate_of):
        result["replicate_of"] = replicate_of
    schema_check(result, "proposal")
    return result


def recommend(request, *, kind="empirical"):
    validate_request(request)
    require(kind in ("empirical", "model", "posterior"), "invalid_spec", "recommendation kind")
    if kind == "posterior":
        from .backends.botorch import posterior_recommendation

        return posterior_recommendation(request)
    if kind == "model":
        try:
            from .backends.botorch import model_recommendation
        except ImportError as exc:  # pragma: no cover - import is lazy inside
            raise Error("missing_dependency", "install bo-craft[botorch]") from exc
        return model_recommendation(request)
    usable = [o for o in request.get("observations", []) if o["status"] == "completed"]
    if "fidelity" in request["problem"]:
        fidelity = request["problem"]["fidelity"]
        usable = [o for o in usable if o["x"][fidelity["parameter"]] == fidelity["target"]]
    usable = [
        o
        for o in usable
        if all(o["values"][c["name"]] <= 0 for c in request["problem"].get("constraints", []))
    ]
    require(usable, "no_observations", "no completed empirically feasible observations")
    objectives = request["problem"]["objectives"]

    def metrics(o):
        return [o["values"][m["name"]] * (1 if m["direction"] == "minimize" else -1) for m in objectives]

    if len(objectives) == 1:
        selected = [min(usable, key=lambda o: metrics(o)[0])]
    else:
        selected = [
            a
            for a in usable
            if not any(
                all(v <= u for v, u in zip(metrics(b), metrics(a)))
                and any(v < u for v, u in zip(metrics(b), metrics(a)))
                for b in usable
            )
        ]
    return {
        "schema_version": "bo.recommendation/0.1",
        "kind": "empirical",
        "observations": selected,
        "evidence": "historical observations; fresh independent validation required",
    }

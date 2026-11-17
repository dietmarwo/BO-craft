"""Strict JSON contracts and scientific observation semantics."""

import hashlib
import json
import math
from pathlib import Path

import jsonschema


class Error(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise Error(code, message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise Error("invalid_json", str(exc)) from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    data = Path(path).read_bytes()
    require(len(data) <= 16 * 1024**2, "size_limit", "JSON input exceeds 16 MiB")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "invalid_json", f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise Error("invalid_json", str(exc)) from exc
    canonical(value)
    return value


def schema_check(value, name):
    schema = json.loads((Path(__file__).parent / "schemas" / f"{name}-0.1.schema.json").read_text())
    try:
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.ValidationError as exc:
        raise Error("invalid_spec", f"{list(exc.absolute_path)}: {exc.message}") from exc


def active(parameter, x):
    return all(x.get(k) in values for k, values in parameter.get("active_if", {}).items())


def validate_point(problem, x):
    require(isinstance(x, dict), "invalid_point", "named parameter object required")
    expected = set()
    for parameter in problem["parameters"]:
        name = parameter["name"]
        if not active(parameter, x):
            continue
        expected.add(name)
        require(name in x, "invalid_point", f"missing active parameter {name}")
        value = x[name]
        kind = parameter["type"]
        if kind == "categorical":
            require(value in parameter["choices"], "invalid_point", f"unknown category {name}")
        else:
            require(
                isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                "invalid_point",
                f"finite number required: {name}",
            )
            require(parameter["lower"] <= value <= parameter["upper"], "invalid_point", f"bounds: {name}")
            if kind == "integer":
                require(isinstance(value, int), "invalid_point", f"integer required: {name}")
    require(set(x) == expected, "invalid_point", "unknown or inactive parameters")
    for constraint in problem.get("input_constraints", []):
        total = sum(coef * x.get(k, 0) for k, coef in constraint["coefficients"].items())
        require(
            total <= constraint["upper"] + constraint.get("tolerance", 0),
            "input_infeasible",
            "known input constraint violated",
        )
    return x


def validate_problem(problem):
    schema_check(problem, "problem")
    parameters = problem["parameters"]
    names = [p["name"] for p in parameters]
    require(len(set(names)) == len(names), "invalid_spec", "duplicate parameter names")
    previous = {}
    for p in parameters:
        if p["type"] == "categorical":
            require(
                "choices" in p and len({canonical(x) for x in p["choices"]}) == len(p["choices"]),
                "invalid_spec",
                "nonempty unique choices required",
            )
            require(not any(k in p for k in ["lower", "upper", "log"]), "invalid_spec", "categorical bounds")
        else:
            require(
                "lower" in p and "upper" in p and p["lower"] < p["upper"],
                "invalid_spec",
                "ordered finite numeric bounds",
            )
            require("choices" not in p, "invalid_spec", "numeric choices")
            if p["type"] == "integer":
                require(
                    isinstance(p["lower"], int) and isinstance(p["upper"], int),
                    "invalid_spec",
                    "integer bounds",
                )
            require(not p.get("log") or p["lower"] > 0, "invalid_spec", "positive log bounds")
        for parent, values in p.get("active_if", {}).items():
            require(
                parent in previous and previous[parent]["type"] == "categorical",
                "invalid_spec",
                "activation parents must be earlier categorical parameters",
            )
            require(
                all(v in previous[parent]["choices"] for v in values),
                "invalid_spec",
                "activation references unknown category",
            )
        previous[p["name"]] = p
    metrics = [m["name"] for m in problem["objectives"] + problem.get("constraints", [])]
    require(len(set(metrics)) == len(metrics), "invalid_spec", "duplicate metric names")
    for c in problem.get("input_constraints", []):
        require(set(c["coefficients"]) <= set(names), "invalid_spec", "unknown constraint parameter")
        require(
            all(
                previous[k]["type"] != "categorical" and not previous[k].get("active_if")
                for k in c["coefficients"]
            ),
            "unsupported_combination",
            "linear constraints need active numeric parameters",
        )
    if "fidelity" in problem:
        f = problem["fidelity"]
        require(f["parameter"] in previous, "invalid_spec", "unknown fidelity parameter")
        p = previous[f["parameter"]]
        require(
            p["type"] == "continuous" and not p.get("active_if") and not p.get("log"),
            "unsupported_combination",
            "fidelity must be active linear continuous",
        )
        require(p["lower"] <= f["target"] <= p["upper"], "invalid_spec", "fidelity target outside bounds")
    canonical(problem)
    return problem


def validate_request(request):
    schema_check(request, "request")
    canonical(request)
    problem = validate_problem(request["problem"])
    seen = set()
    required = {m["name"] for m in problem["objectives"] + problem.get("constraints", [])}
    for item in request.get("observations", []):
        require(item["id"] not in seen, "invalid_observation", "duplicate observation ID")
        seen.add(item["id"])
        validate_point(problem, item["x"])
        if item["status"] == "completed":
            require(
                set(item.get("values", {})) == required, "invalid_observation", "all named metrics required"
            )
            variances = item.get("variances", {})
            require(set(variances) <= required, "invalid_observation", "unknown metric variance")
            if item.get("count", 1) > 1:
                require(
                    item.get("variance_kind") == "mean",
                    "invalid_observation",
                    "aggregated observations require variance of the estimated mean",
                )
        else:
            require(
                not item.get("values") and not item.get("variances"),
                "invalid_observation",
                "noncompleted outcomes are not numerical training targets",
            )
    pending_ids = set()
    for item in request.get("pending", []):
        require(
            item["id"] not in pending_ids and item["id"] not in seen,
            "invalid_observation",
            "duplicate pending ID",
        )
        pending_ids.add(item["id"])
        validate_point(problem, item["x"])
    require(len(request.get("observations", [])) <= 10000, "size_limit", "archive cap 10000")
    return request


def aggregate(replicates, identity):
    require(len(replicates) >= 2, "invalid_observation", "at least two independent replicates")
    seeds = [r["seed"] for r in replicates]
    require(len(set(seeds)) == len(seeds), "invalid_observation", "independent replicate seeds required")
    first = replicates[0]
    require(
        all(
            r["status"] == "completed" and r["x"] == first["x"] and set(r["values"]) == set(first["values"])
            for r in replicates
        ),
        "invalid_observation",
        "matching completed replicates required",
    )
    n = len(replicates)
    values = {key: sum(r["values"][key] for r in replicates) / n for key in first["values"]}
    variances = {
        key: sum((r["values"][key] - mean) ** 2 for r in replicates) / (n - 1) / n
        for key, mean in values.items()
    }
    return {
        "id": identity,
        "trial_id": first["trial_id"],
        "x": first["x"],
        "status": "completed",
        "values": values,
        "variances": variances,
        "count": n,
        "variance_kind": "mean",
        "replicate_ids": [r["id"] for r in replicates],
    }


def encode(problem, x):
    validate_point(problem, x)
    result = []
    for p in problem["parameters"]:
        require(
            p["type"] == "continuous" and not p.get("active_if"),
            "unsupported_combination",
            "BoTorch adapter requires continuous active parameters",
        )
        lo, hi, value = p["lower"], p["upper"], x[p["name"]]
        if p.get("log"):
            lo, hi, value = math.log(lo), math.log(hi), math.log(value)
        result.append((value - lo) / (hi - lo))
    return result


def decode(problem, unit):
    result = {}
    for p, value in zip(problem["parameters"], unit, strict=True):
        lo, hi = p["lower"], p["upper"]
        if p.get("log"):
            value = math.exp(math.log(lo) + float(value) * (math.log(hi) - math.log(lo)))
        else:
            value = lo + float(value) * (hi - lo)
        result[p["name"]] = min(hi, max(lo, value))
    return validate_point(problem, result)

"""Bounded CPU BoTorch adapter.

Inputs are encoded in a declared model space: continuous parameters on [0, 1]
(log-scaled when declared), integers as their allowed normalized values,
categoricals as category indices. Metric values stay in raw units apart from
an optional, declared monotone transform of the objectives.
"""

import math
import warnings

from ..specs import Error, canonical, require

DEFAULTS = {
    "initial_points": 8,
    "fit_maxiter": 50,
    # 0.1.0's acquisition settings; rule D1 of the 17 November protocol kept them
    # (docs/evidence/plan-2026-11-17). acq_maxiter was fit_maxiter's value before.
    "acq_maxiter": 50,
    "num_restarts": 4,
    "raw_samples": 128,
    "mc_samples": 128,
    "training_cap": 256,
    "feasibility_probability": 0.95,
    "replication_threshold": 0.5,
    "max_replicates": 4,
    "num_fantasies": 16,
}
# Integers with more allowed values are refused; use Optuna for them.
MAX_INTEGER_VALUES = 4096
FOREIGN = ["group", "multivariate", "constant_liar", "service", "sampler"]


def setting(config, key):
    return config.get(key, DEFAULTS[key])


class Space:
    """The model space of a flat (unconditional) parameter list."""

    def __init__(self, problem):
        self.parameters = problem["parameters"]
        require(
            not any(p.get("active_if") for p in self.parameters),
            "unsupported_combination",
            "BoTorch supports unconditional parameters; use Optuna TPE for conditional spaces",
        )
        self.discrete = {}
        self.categorical = {}
        for i, p in enumerate(self.parameters):
            if p["type"] == "integer":
                count = p["upper"] - p["lower"] + 1
                require(
                    count <= MAX_INTEGER_VALUES,
                    "unsupported_combination",
                    f"integer {p['name']} has more than {MAX_INTEGER_VALUES} values; use Optuna",
                )
                self.discrete[i] = [self.unit(p, v) for v in range(p["lower"], p["upper"] + 1)]
            elif p["type"] == "categorical":
                self.categorical[i] = [float(k) for k in range(len(p["choices"]))]

    @property
    def mixed(self):
        return bool(self.discrete or self.categorical)

    @property
    def dimension(self):
        return len(self.parameters)

    @staticmethod
    def unit(p, value):
        lo, hi = p["lower"], p["upper"]
        if p.get("log"):
            lo, hi, value = math.log(lo), math.log(hi), math.log(value)
        return (value - lo) / (hi - lo)

    def bounds(self):
        import torch

        upper = [
            len(self.categorical[i]) - 1.0 if i in self.categorical else 1.0 for i in range(self.dimension)
        ]
        return torch.tensor([[0.0] * self.dimension, upper], dtype=torch.double)

    def encode(self, x):
        row = []
        for p in self.parameters:
            if p["type"] == "categorical":
                row.append(float(p["choices"].index(x[p["name"]])))
            else:
                row.append(self.unit(p, x[p["name"]]))
        return row

    def decode(self, row):
        point = {}
        for p, value in zip(self.parameters, row, strict=True):
            value = float(value)
            if p["type"] == "categorical":
                index = min(len(p["choices"]) - 1, max(0, round(value)))
                point[p["name"]] = p["choices"][index]
                continue
            lo, hi = p["lower"], p["upper"]
            if p.get("log"):
                raw = math.exp(math.log(lo) + value * (math.log(hi) - math.log(lo)))
            else:
                raw = lo + value * (hi - lo)
            if p["type"] == "integer":
                raw = round(raw)
            point[p["name"]] = min(hi, max(lo, raw))
        return point


def validate(request):
    """The acquisition for a supported request; unsupported combinations fail first."""
    p, c = request["problem"], request["config"]
    require(
        not any(k in c for k in FOREIGN),
        "unsupported_combination",
        "selected configuration belongs to another backend",
    )
    space = Space(p)
    require(
        not p.get("input_constraints") or not any(v.get("log") for v in p["parameters"]),
        "unsupported_combination",
        "linear input constraints with log parameters require a parameterization",
    )
    scalar = len(p["objectives"]) == 1
    if p.get("fidelity"):
        require(
            scalar and not p.get("constraints") and not request.get("pending") and not space.mixed,
            "unsupported_combination",
            "multifidelity currently scalar, unconstrained, continuous and synchronous",
        )
    acquisition = c.get("acquisition")
    expected = "multifidelity_kg" if p.get("fidelity") else ("log_nei" if scalar else "log_nehvi")
    require(acquisition in (None, expected), "unsupported_combination", f"expected acquisition {expected}")
    if not scalar:
        require(
            len(c.get("reference_point", [])) == len(p["objectives"]),
            "invalid_spec",
            "one raw-unit reference value per objective",
        )
    if c.get("trust_region"):
        require(
            scalar
            and not p.get("constraints")
            and not p.get("fidelity")
            and not p.get("input_constraints")
            and not space.mixed,
            "unsupported_combination",
            "trust_region: scalar unconstrained continuous problems only",
        )
    if c.get("replication", "none") != "none":
        require(
            scalar and not p.get("fidelity") and c.get("noise", "inferred") != "deterministic",
            "unsupported_combination",
            "replication: scalar noisy problems without fidelity only",
        )
    if c.get("outcome_transform", "standardize") != "standardize":
        require(
            not p.get("fidelity"),
            "unsupported_combination",
            "outcome transforms with multifidelity are not validated",
        )
        require(
            c["outcome_transform"] != "log" or scalar or all(v > 0 for v in c["reference_point"]),
            "invalid_spec",
            "outcome_transform log needs positive reference values",
        )
    return expected


# --- training data -------------------------------------------------------------


def signed_objectives(problem, values):
    """Objective values oriented so that larger is better."""
    return [values[m["name"]] * (1 if m["direction"] == "maximize" else -1) for m in problem["objectives"]]


def violation(problem, values):
    return sum(max(0.0, values[c["name"]]) for c in problem.get("constraints", []))


def dominated(a, b):
    """True if b dominates a (larger is better)."""
    return all(v <= u for v, u in zip(a, b)) and any(v < u for v, u in zip(a, b))


def select_training(problem, data, cap):
    """The best quarter of the cap plus the latest observations, in ledger order."""
    if len(data) <= cap:
        return data, "all completed observations"
    k = cap // 4
    feasible = [o for o in data if violation(problem, o["values"]) == 0]
    if len(problem["objectives"]) == 1:
        ranked = sorted(feasible, key=lambda o: -signed_objectives(problem, o["values"])[0])
    else:
        scores = [signed_objectives(problem, o["values"]) for o in feasible]
        ranked = [o for o, s in zip(feasible, scores) if not any(dominated(s, t) for t in scores)]
    infeasible = sorted(
        (o for o in data if violation(problem, o["values"]) > 0),
        key=lambda o: violation(problem, o["values"]),
    )
    best = {o["id"] for o in (ranked + infeasible)[:k]}
    recent = [o for o in data if o["id"] not in best][-(cap - len(best)) :]
    keep = best | {o["id"] for o in recent}
    return [o for o in data if o["id"] in keep], f"best {len(best)} plus latest {len(recent)}"


TRANSFORMS = {
    "standardize": (lambda v: v, lambda d: 1.0, lambda t: t),
    "log": (math.log, lambda v: 1.0 / v, math.exp),
    "bilog": (
        lambda v: math.copysign(math.log1p(abs(v)), v),
        lambda v: 1.0 / (1.0 + abs(v)),
        lambda t: math.copysign(math.expm1(abs(t)), t),
    ),
}


class Fit:
    pass


def fit_model(problem, observations, config, seed):
    """A fitted GP over the selected completed observations of a request."""
    import torch
    from botorch.exceptions.errors import ModelFittingError
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MixedSingleTaskGP, SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    torch.set_num_threads(config.get("threads", 1))
    torch.manual_seed(seed)
    space = Space(problem)
    data = [o for o in observations if o["status"] == "completed"]
    require(len(data) >= 2, "no_observations", "a model needs two completed observations")
    data, rule = select_training(problem, data, setting(config, "training_cap"))
    objectives = problem["objectives"]
    names = [m["name"] for m in objectives + problem.get("constraints", [])]
    signs = [1 if m["direction"] == "maximize" else -1 for m in objectives]
    signs += [1] * len(problem.get("constraints", []))
    transform_name = config.get("outcome_transform", "standardize")
    forward, slope, _ = TRANSFORMS[transform_name]
    transformed = {m["name"] for m in objectives} if transform_name != "standardize" else set()
    if transform_name == "log":
        require(
            all(o["values"][k] > 0 for o in data for k in transformed),
            "invalid_observation",
            "outcome_transform log needs positive objective values",
        )

    def target(o, key):
        return forward(o["values"][key]) if key in transformed else o["values"][key]

    x = torch.tensor([space.encode(o["x"]) for o in data], dtype=torch.double)
    y = torch.tensor([[target(o, k) * s for k, s in zip(names, signs)] for o in data], dtype=torch.double)
    noise = config.get("noise", "inferred")
    yvar = None
    if noise == "known":
        require(
            all(set(o.get("variances", {})) == set(names) for o in data),
            "invalid_observation",
            "known-noise mode needs variance of each metric observation",
        )
        yvar = torch.tensor(
            [
                [o["variances"][k] * (slope(o["values"][k]) ** 2 if k in transformed else 1.0) for k in names]
                for o in data
            ],
            dtype=torch.double,
        ).clamp_min(1e-12)
    elif noise == "deterministic":
        yvar = torch.full_like(y, 1e-10)
    kwargs = {"train_X": x, "train_Y": y, "train_Yvar": yvar, "outcome_transform": Standardize(m=len(names))}
    if problem.get("fidelity"):
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

        index = [v["name"] for v in problem["parameters"]].index(problem["fidelity"]["parameter"])
        model = SingleTaskMultiFidelityGP(**kwargs, data_fidelities=[index])
    elif space.categorical:
        model = MixedSingleTaskGP(**kwargs, cat_dims=sorted(space.categorical))
    else:
        model = SingleTaskGP(**kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": setting(config, "fit_maxiter")}}, max_attempts=1
            )
        except (ModelFittingError, RuntimeError, ValueError) as exc:
            # A typed failure, never a silent fallback to another design.
            raise Error("model_fit_failed", f"GP hyperparameter fit failed: {str(exc)[:1024]}") from exc
    model.eval()
    fit = Fit()
    fit.model, fit.x, fit.data, fit.space = model, x, data, space
    fit.names, fit.signs, fit.transform, fit.transformed = names, signs, transform_name, transformed
    fit.selection = rule
    fit.warnings = [str(w.message)[:512] for w in caught][:8]
    fit.diagnostics = diagnostics(model, names, fit.warnings)
    return fit


def model_for(request):
    """Compatibility wrapper: (model, training inputs, selected observations, warnings)."""
    fit = fit_model(request["problem"], request.get("observations", []), request["config"], request["seed"])
    return fit.model, fit.x, fit.data, fit.warnings


def diagnostics(model, names, messages):
    """Fitted hyperparameters and exact leave-one-out residuals, in standardized units."""
    import torch
    from gpytorch.kernels import Kernel
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    def rounded(values):
        return [float(f"{v:.4g}") for v in values]

    result = {"metrics": names, "lengthscales": {}, "warnings": []}
    with torch.no_grad():
        for name, module in model.covar_module.named_modules():
            if isinstance(module, Kernel) and module.has_lengthscale and module.lengthscale is not None:
                ls = module.lengthscale.detach().reshape(max(1, len(names)), -1)
                result["lengthscales"][name or "kernel"] = [rounded(row.tolist()) for row in ls]
                bound = getattr(module.raw_lengthscale_constraint, "lower_bound", None)
                if bound is not None and bool((ls <= 1.05 * float(bound)).any()):
                    result["warnings"].append(f"a lengthscale of {name or 'kernel'} is at its lower bound")
        fixed = isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
        if not fixed:
            result["noise_variance"] = rounded(model.likelihood.noise.detach().reshape(-1).tolist())
        x = model.train_inputs[0]
        y = model.train_targets
        k = model.covar_module(x).to_dense()
        noise = model.likelihood.noise.detach()
        if noise.shape[-1] != y.shape[-1]:
            noise = noise.expand(y.shape)
        k = k + torch.diag_embed(noise.reshape(y.shape))
        residual = (y - model.mean_module(x)).unsqueeze(-1)
        try:
            chol = torch.linalg.cholesky(k + 1e-9 * torch.eye(k.shape[-1], dtype=k.dtype))
        except RuntimeError:
            result["warnings"].append("leave-one-out check skipped: covariance not positive definite")
            return result
        inverse = torch.cholesky_inverse(chol)
        alpha = (inverse @ residual).squeeze(-1)
        diagonal = torch.diagonal(inverse, dim1=-2, dim2=-1)
        loo = (alpha / diagonal).reshape(-1, y.shape[-1])
        z = (alpha / diagonal.sqrt()).reshape(-1, y.shape[-1])
        result["loo_rmse"] = rounded(loo.pow(2).mean(-1).sqrt().tolist())
        result["loo_rms_z"] = rounded(z.pow(2).mean(-1).sqrt().tolist())
        result["loo_max_abs_z"] = rounded(z.abs().max(-1).values.tolist())
        for metric, rms in zip(names, result["loo_rms_z"]):
            if rms > 2:
                result["warnings"].append(f"{metric}: leave-one-out errors exceed the model's uncertainty")
    result["warnings"] += [m for m in messages if "converge" in m.lower()][:2]
    return result


# --- posterior summaries ------------------------------------------------------------


def posterior_rows(fit, problem, rows):
    """Posterior means and latent standard deviations of every metric at encoded rows."""
    import torch

    with torch.no_grad():
        posterior = fit.model.posterior(torch.tensor(rows, dtype=torch.double))
        mean = posterior.mean.reshape(len(rows), -1)
        sd = posterior.variance.clamp_min(0).sqrt().reshape(len(rows), -1)
    return mean, sd


def feasibility(fit, problem, mean, sd):
    import torch

    n = len(problem["objectives"])
    p = torch.ones(mean.shape[0], dtype=torch.double)
    for j in range(len(problem.get("constraints", []))):
        normal = torch.distributions.Normal(mean[:, n + j], sd[:, n + j].clamp_min(1e-12))
        p = p * normal.cdf(torch.zeros(mean.shape[0], dtype=torch.double))
    return p


def raw_summary(fit, problem, mean, sd, index):
    """Back-transformed posterior summary of each objective at one row."""
    _, _, inverse = TRANSFORMS[fit.transform]
    result = {}
    for j, m in enumerate(problem["objectives"]):
        sign = 1 if m["direction"] == "maximize" else -1
        t, s = sign * float(mean[index, j]), float(sd[index, j])
        back = inverse if m["name"] in fit.transformed else (lambda v: v)
        low, high = sorted([back(t - 1.96 * s), back(t + 1.96 * s)])
        entry = {"median": back(t), "interval_95": [low, high], "latent_sd_model_units": s}
        if m["name"] not in fit.transformed:
            entry["mean"] = t
        result[m["name"]] = entry
    return result


def incumbent(fit, problem, candidates, config):
    """The model's choice among candidate observations: (rows, means, sds, eligible, chosen)."""
    rows = [fit.space.encode(o["x"]) for o in candidates]
    mean, sd = posterior_rows(fit, problem, rows)
    p = feasibility(fit, problem, mean, sd)
    eligible = [
        i for i in range(len(candidates)) if float(p[i]) >= setting(config, "feasibility_probability")
    ]
    n = len(problem["objectives"])
    if not eligible:
        return mean, sd, p, []
    if n == 1:
        chosen = [max(eligible, key=lambda i: float(mean[i, 0]))]
    else:
        scores = {i: mean[i, :n].tolist() for i in eligible}
        chosen = [i for i in eligible if not any(dominated(scores[i], scores[j]) for j in eligible)]
    return mean, sd, p, chosen


def model_recommendation(request):
    """Observed points chosen by the posterior mean; constrained and multiobjective included."""
    p, c = request["problem"], request["config"]
    Space(p)
    completed = [o for o in request.get("observations", []) if o["status"] == "completed"]
    candidates = completed
    if "fidelity" in p:
        f = p["fidelity"]
        candidates = [o for o in completed if o["x"][f["parameter"]] == f["target"]]
    require(candidates, "no_observations", "no completed candidate observations")
    fit = fit_model(p, completed, c, request["seed"])
    mean, sd, feasible, chosen = incumbent(fit, p, candidates, c)
    require(
        chosen,
        "no_feasible_recommendation",
        f"no observed point has feasibility probability >= {setting(c, 'feasibility_probability')}",
    )
    groups = {}
    for i in chosen:
        groups.setdefault(canonical(candidates[i]["x"]), []).append(i)
    points = []
    for indices in groups.values():
        i = indices[0]
        same = [o["id"] for o in candidates if canonical(o["x"]) == canonical(candidates[i]["x"])]
        points.append(
            {
                "x": candidates[i]["x"],
                "observation_ids": same,
                "objectives": raw_summary(fit, p, mean, sd, i),
                "probability_feasible": float(feasible[i]),
            }
        )
    return {
        "schema_version": "bo.recommendation/0.1",
        "kind": "model",
        "points": points,
        "observations": [candidates[i] for i in chosen],
        "uncertainty": "latent GP conditional on fitted hyperparameters; no frequentist calibration claim",
        "evidence": "observed points ranked by a fitted model; values are predictions; "
        "fresh independent validation required",
        "outcome_transform": fit.transform,
        "training_ids": [o["id"] for o in fit.data],
        "training_selection": fit.selection,
        "diagnostics": fit.diagnostics,
    }


def posterior_recommendation(request):
    """Scalar latent-median recommendation over the whole space, conditional on the fitted GP.

    This is a prediction artifact and never carries native evaluated evidence.
    """
    require(
        request["config"]["backend"] == "botorch", "unsupported_combination", "posterior requires BoTorch"
    )
    validate(request)
    p, c = request["problem"], request["config"]
    require(
        len(p["objectives"]) == 1 and not p.get("constraints") and not p.get("input_constraints"),
        "unsupported_combination",
        "posterior recommendation currently unconstrained scalar",
    )
    require(
        sum(o["status"] == "completed" for o in request.get("observations", [])) >= 2,
        "no_observations",
        "posterior recommendation requires two observations",
    )
    from botorch.acquisition.analytic import PosteriorMean
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

    fit = fit_model(p, request.get("observations", []), c, request["seed"])
    space, d = fit.space, fit.space.dimension
    acquisition = PosteriorMean(fit.model)
    bounds = space.bounds()
    fixed = None
    if "fidelity" in p:
        f = p["fidelity"]
        index = [v["name"] for v in p["parameters"]].index(f["parameter"])
        target = Space.unit(p["parameters"][index], f["target"])
        acquisition = FixedFeatureAcquisitionFunction(acquisition, d=d, columns=[index], values=[target])
        fixed = (index, target)
        keep = [i for i in range(d) if i != index]
        require(keep, "unsupported_combination", "at least one design coordinate")
        bounds = bounds[:, keep]
    x = optimize(acquisition, space, bounds, 1, c, reduced=fixed is not None)[0]
    if fixed:
        import torch

        index, target = fixed
        full = torch.empty(1, d, dtype=torch.double)
        full[0, index] = target
        full[0, [i for i in range(d) if i != index]] = x[0]
        x = full
    point = space.decode(x[0].tolist())
    mean, sd = posterior_rows(fit, p, [space.encode(point)])
    summary = raw_summary(fit, p, mean, sd, 0)[p["objectives"][0]["name"]]
    result = {
        "schema_version": "bo.recommendation/0.1",
        "kind": "posterior",
        "x": point,
        "metric": p["objectives"][0]["name"],
        "median": summary["median"],
        "latent_standard_deviation": summary["latent_sd_model_units"],
        "pointwise_model_interval": summary["interval_95"],
        "uncertainty": "latent GP conditional on fitted hyperparameters; no frequentist calibration claim",
        "evidence": "predicted; fresh independent evaluation required",
        "outcome_transform": fit.transform,
        "training_ids": [o["id"] for o in fit.data],
        "warnings": fit.warnings,
        "diagnostics": fit.diagnostics,
    }
    if "mean" in summary:
        result["mean"] = summary["mean"]
    return result


# --- acquisition --------------------------------------------------------------------


def optimize(acquisition, space, bounds, q, config, *, reduced=False, inequality=None, sequential=True):
    """Acquisition optimization in the model space; mixed spaces use alternating search."""
    from botorch.optim import optimize_acqf

    options = {"batch_limit": 2}
    if space.mixed and not reduced:
        from botorch.optim import optimize_acqf_mixed_alternating

        points, value = optimize_acqf_mixed_alternating(
            acquisition,
            bounds,
            discrete_dims=space.discrete or None,
            cat_dims=space.categorical or None,
            q=q,
            num_restarts=setting(config, "num_restarts"),
            raw_samples=setting(config, "raw_samples"),
            options=options,
            inequality_constraints=inequality,
            sequential=sequential,
        )
        return points, value
    options["maxiter"] = setting(config, "acq_maxiter")
    return optimize_acqf(
        acquisition,
        bounds,
        q,
        num_restarts=setting(config, "num_restarts"),
        raw_samples=setting(config, "raw_samples"),
        inequality_constraints=inequality,
        options=options,
        sequential=sequential,
    )


def trust_region_state(problem, data, dimension, initial):
    """TuRBO-1 side length replayed from the ledger; no hidden optimizer state."""
    length, successes, failures, restarts = 0.8, 0, 0, 0
    fail_tol, succ_tol = max(4, dimension), 3
    best = -math.inf
    for i, o in enumerate(data):
        y = signed_objectives(problem, o["values"])[0]
        if i >= initial:
            if y > best + 1e-3 * abs(best):
                successes, failures = successes + 1, 0
            else:
                successes, failures = 0, failures + 1
            if successes == succ_tol:
                length, successes = min(2 * length, 1.6), 0
            elif failures == fail_tol:
                length, failures = length / 2, 0
            if length < 0.5**7:
                length, restarts = 0.8, restarts + 1
        best = max(best, y)
    return length, restarts


def inequality_constraints(problem):
    import torch

    result = []
    for con in problem.get("input_constraints", []):
        coef, idx, offset = [], [], 0
        for i, parameter in enumerate(problem["parameters"]):
            a = con["coefficients"].get(parameter["name"], 0)
            if a:
                idx.append(i)
                coef.append(-a * (parameter["upper"] - parameter["lower"]))
                offset += a * parameter["lower"]
        result.append(
            (
                torch.tensor(idx),
                torch.tensor(coef, dtype=torch.double),
                offset - con["upper"] - con.get("tolerance", 0),
            )
        )
    return result or None


def replication_decision(fit, problem, request):
    """Re-evaluate the model's incumbent while its latent uncertainty exceeds the noise level."""
    import torch

    c = request["config"]
    completed = fit.data
    _, sd, _, chosen = incumbent(fit, problem, completed, c)
    if not chosen:
        return None, {"policy": "incumbent", "replicated": False, "reason": "no eligible incumbent"}
    best = completed[chosen[0]]
    key = canonical(best["x"])
    evaluations = sum(canonical(o["x"]) == key for o in request.get("observations", []))
    pending = any(canonical(o["x"]) == key for o in request.get("pending", []))
    stdv = float(fit.model.outcome_transform.stdvs.reshape(-1)[0])
    if c.get("noise", "inferred") == "known":
        rows = [i for i, o in enumerate(completed) if canonical(o["x"]) == key]
        with torch.no_grad():
            noise_var = float(fit.model.likelihood.noise.reshape(-1)[rows].mean()) * stdv**2
    else:
        noise_var = float(fit.model.likelihood.noise.reshape(-1)[0]) * stdv**2
    latent_sd = float(sd[chosen[0], 0])
    noise_sd = math.sqrt(max(noise_var, 0.0))
    replicate = (
        not pending
        and evaluations < setting(c, "max_replicates")
        and latent_sd > setting(c, "replication_threshold") * noise_sd
    )
    record = {
        "policy": "incumbent",
        "incumbent_id": best["id"],
        "latent_sd": latent_sd,
        "noise_sd": noise_sd,
        "evaluations_at_incumbent": evaluations,
        "replicated": replicate,
    }
    return (best if replicate else None), record


def propose(request):
    expected = validate(request)  # no heavy imports for unsupported combinations
    from ..api import random_points

    c = request["config"]
    data = [o for o in request.get("observations", []) if o["status"] == "completed"]
    if len(data) < setting(c, "initial_points"):
        return random_points(request), {
            "phase": "initial_design",
            "design": "seeded random",
            "acquisition": expected,
        }
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.sampling.normal import SobolQMCNormalSampler
    except ImportError as exc:
        raise Error("missing_dependency", "install bo-craft[botorch] and a CPU Torch build") from exc
    p = request["problem"]
    fit = fit_model(p, request.get("observations", []), c, request["seed"])
    model, x, space = fit.model, fit.x, fit.space
    d, nobj, q = space.dimension, len(p["objectives"]), request["q"]
    diagnostic = {
        "phase": "model",
        "acquisition": expected,
        "noise": c.get("noise", "inferred"),
        "outcome_transform": fit.transform,
        "training_ids": [o["id"] for o in fit.data],
        "training_selection": fit.selection,
        "warnings": fit.warnings,
        "model": fit.diagnostics,
        "space": "mixed" if space.mixed else "continuous",
        "precision": "float64",
        "device": "cpu",
    }
    replicate = None
    if c.get("replication", "none") == "incumbent":
        replicate, diagnostic["replication"] = replication_decision(fit, p, request)
    pending_rows = [space.encode(o["x"]) for o in request.get("pending", [])]
    if replicate is not None:
        pending_rows.append(space.encode(replicate["x"]))
    replicate_of = [replicate["id"]] if replicate is not None else []
    q_model = q - len(replicate_of)
    candidates = [replicate["x"]] if replicate is not None else []
    if q_model == 0:
        diagnostic["replicate_of"] = replicate_of
        return candidates, diagnostic
    x_pending = torch.tensor(pending_rows, dtype=torch.double) if pending_rows else None
    constraints = [(lambda z, index=nobj + i: z[..., index]) for i, _ in enumerate(p.get("constraints", []))]
    sampler = SobolQMCNormalSampler(torch.Size([setting(c, "mc_samples")]), seed=request["seed"])
    common = {
        "model": model,
        "X_baseline": x,
        "sampler": sampler,
        "X_pending": x_pending,
        "constraints": constraints or None,
    }
    bounds = space.bounds()
    if c.get("trust_region"):
        length, restarts = trust_region_state(p, data, d, setting(c, "initial_points"))
        center_obs = max(data, key=lambda o: signed_objectives(p, o["values"])[0])
        center = torch.tensor(space.encode(center_obs["x"]), dtype=torch.double)
        ls = (
            next(k.lengthscale for k in model.covar_module.modules() if getattr(k, "has_lengthscale", False))
            .detach()
            .reshape(-1)
        )
        weights = ls / ls.log().mean().exp()
        bounds = torch.stack(
            [
                (center - weights * length / 2).clamp(0.0, 1.0),
                (center + weights * length / 2).clamp(0.0, 1.0),
            ]
        )
        diagnostic["trust_region"] = {
            "side_length": length,
            "restarts": restarts,
            "center_id": center_obs["id"],
        }
    if expected == "log_nei":
        acq = qLogNoisyExpectedImprovement(
            **common, objective=GenericMCObjective(lambda z, X=None: z[..., 0]), prune_baseline=True
        )
    elif expected == "log_nehvi":
        ref = [
            v * (1 if m["direction"] == "maximize" else -1)
            for m, v in zip(p["objectives"], c["reference_point"])
        ]
        if fit.transformed:
            forward = TRANSFORMS[fit.transform][0]
            ref = [
                (1 if m["direction"] == "maximize" else -1) * forward(v)
                if m["name"] in fit.transformed
                else r
                for m, v, r in zip(p["objectives"], c["reference_point"], ref)
            ]
        acq = qLogNoisyExpectedHypervolumeImprovement(
            **common,
            ref_point=ref,
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(nobj))),
            prune_baseline=True,
        )
    else:
        acq = knowledge_gradient(model, p, c, space)
    try:
        points, value = optimize(
            acq,
            space,
            bounds,
            q_model,
            c,
            inequality=inequality_constraints(p),
            sequential=expected != "multifidelity_kg",
        )
    except (RuntimeError, ValueError) as exc:
        raise Error("numerical_failure", str(exc)[:1024]) from exc
    candidates += [space.decode(row.tolist()) for row in points.detach()]
    diagnostic["acquisition_value"] = value.detach().reshape(-1).tolist() if value is not None else []
    if replicate_of:
        diagnostic["replicate_of"] = replicate_of + [None] * q_model
    return candidates, diagnostic


def knowledge_gradient(model, p, c, space):
    import torch
    from botorch.acquisition.analytic import PosteriorMean
    from botorch.acquisition.cost_aware import InverseCostWeightedUtility
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
    from botorch.acquisition.utils import project_to_target_fidelity
    from botorch.models.cost import AffineFidelityCostModel

    d = space.dimension
    f = p["fidelity"]
    index = [v["name"] for v in p["parameters"]].index(f["parameter"])
    target = Space.unit(p["parameters"][index], f["target"])
    # Cost is defined on normalized fidelity; positivity is validated in specs.
    cost = AffineFidelityCostModel(fidelity_weights={index: f["cost_weight"]}, fixed_cost=f["cost_intercept"])
    require(d >= 2, "unsupported_combination", "multifidelity needs at least one design parameter")
    fixed = FixedFeatureAcquisitionFunction(PosteriorMean(model), d=d, columns=[index], values=[target])
    reduced = torch.stack([torch.zeros(d - 1, dtype=torch.double), torch.ones(d - 1, dtype=torch.double)])
    _, current = optimize(fixed, space, reduced, 1, c, reduced=True)
    return qMultiFidelityKnowledgeGradient(
        model,
        num_fantasies=setting(c, "num_fantasies"),
        current_value=current.detach(),
        cost_aware_utility=InverseCostWeightedUtility(cost_model=cost),
        project=lambda z: project_to_target_fidelity(z, d=d, target_fidelities={index: target}),
    )

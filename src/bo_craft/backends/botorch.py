"""Bounded CPU BoTorch adapter; normalized continuous inputs, raw metric units."""

import warnings

from ..specs import Error, decode, encode, require


def validate(request):
    p, c = request["problem"], request["config"]
    require(
        not any(k in c for k in ["group", "multivariate", "constant_liar", "service"]),
        "unsupported_combination",
        "selected configuration belongs to another backend",
    )
    require(
        all(v["type"] == "continuous" and not v.get("active_if") for v in p["parameters"]),
        "unsupported_combination",
        "BoTorch supports active continuous parameters; use Optuna for mixed spaces",
    )
    require(
        not p.get("input_constraints") or not any(v.get("log") for v in p["parameters"]),
        "unsupported_combination",
        "linear input constraints with log parameters require a parameterization",
    )
    if p.get("fidelity"):
        require(
            len(p["objectives"]) == 1 and not p.get("constraints") and not request.get("pending"),
            "unsupported_combination",
            "multifidelity currently scalar, unconstrained and synchronous",
        )
    acquisition = c.get("acquisition")
    expected = (
        "multifidelity_kg" if p.get("fidelity") else ("log_nehvi" if len(p["objectives"]) > 1 else "log_nei")
    )
    require(acquisition in (None, expected), "unsupported_combination", f"expected acquisition {expected}")
    if len(p["objectives"]) > 1:
        require(
            len(c.get("reference_point", [])) == len(p["objectives"]),
            "invalid_spec",
            "one raw-unit reference value per objective",
        )
    return expected


def posterior_recommendation(request):
    """Scalar latent-mean recommendation, conditional on the fitted GP.

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
    import torch
    from botorch.acquisition.analytic import PosteriorMean
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.optim import optimize_acqf

    model, _, selected, messages = model_for(request)
    d = len(p["parameters"])
    acquisition = PosteriorMean(model)
    fixed = None
    if "fidelity" in p:
        f = p["fidelity"]
        index = [v["name"] for v in p["parameters"]].index(f["parameter"])
        parameter = p["parameters"][index]
        target = (f["target"] - parameter["lower"]) / (parameter["upper"] - parameter["lower"])
        acquisition = FixedFeatureAcquisitionFunction(acquisition, d=d, columns=[index], values=[target])
        fixed = (index, target)
    width = d - (fixed is not None)
    require(width >= 1, "unsupported_combination", "at least one design coordinate")
    bounds = torch.stack([torch.zeros(width, dtype=torch.double), torch.ones(width, dtype=torch.double)])
    x, _ = optimize_acqf(
        acquisition,
        bounds,
        1,
        num_restarts=c.get("num_restarts", 4),
        raw_samples=c.get("raw_samples", 128),
        options={"maxiter": c.get("fit_maxiter", 50)},
    )
    if fixed:
        index, target = fixed
        full = torch.empty(1, d, dtype=torch.double)
        full[0, index] = target
        full[0, [i for i in range(d) if i != index]] = x[0]
        x = full
    posterior = model.posterior(x)
    sign = 1 if p["objectives"][0]["direction"] == "maximize" else -1
    mean = sign * float(posterior.mean.item())
    sd = float(posterior.variance.clamp_min(0).sqrt().item())
    return {
        "schema_version": "bo.recommendation/0.1",
        "kind": "posterior",
        "x": decode(p, x[0].detach().tolist()),
        "metric": p["objectives"][0]["name"],
        "mean": mean,
        "latent_standard_deviation": sd,
        "pointwise_model_interval": [mean - 1.96 * sd, mean + 1.96 * sd],
        "uncertainty": "latent GP conditional on fitted hyperparameters; no frequentist calibration claim",
        "evidence": "predicted; fresh independent evaluation required",
        "training_ids": [o["id"] for o in selected],
        "warnings": messages,
    }


def model_for(request):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    p, c = request["problem"], request["config"]
    torch.set_num_threads(c.get("threads", 1))
    torch.manual_seed(request["seed"])
    data = [o for o in request.get("observations", []) if o["status"] == "completed"]
    cap = c.get("training_cap", 256)
    # Explicit deterministic suffix selection; full archive remains caller owned.
    data = data[-cap:]
    names = [m["name"] for m in p["objectives"] + p.get("constraints", [])]
    signs = [1 if m["direction"] == "maximize" else -1 for m in p["objectives"]]
    signs += [1] * len(p.get("constraints", []))
    x = torch.tensor([encode(p, o["x"]) for o in data], dtype=torch.double)
    y = torch.tensor(
        [[o["values"][k] * sign for k, sign in zip(names, signs)] for o in data], dtype=torch.double
    )
    noise = c.get("noise", "inferred")
    yvar = None
    if noise == "known":
        require(
            all(set(o.get("variances", {})) == set(names) for o in data),
            "invalid_observation",
            "known-noise mode needs variance of each metric observation",
        )
        yvar = torch.tensor([[o["variances"][k] for k in names] for o in data], dtype=torch.double).clamp_min(
            1e-12
        )
    elif noise == "deterministic":
        yvar = torch.full_like(y, 1e-10)
    kwargs = {"train_X": x, "train_Y": y, "train_Yvar": yvar, "outcome_transform": Standardize(m=len(names))}
    if p.get("fidelity"):
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

        index = [v["name"] for v in p["parameters"]].index(p["fidelity"]["parameter"])
        model = SingleTaskMultiFidelityGP(**kwargs, data_fidelities=[index])
    else:
        model = SingleTaskGP(**kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    from botorch.exceptions.errors import ModelFittingError

    with warnings.catch_warnings(record=True) as caught:
        try:
            fit_gpytorch_mll(
                mll, optimizer_kwargs={"options": {"maxiter": c.get("fit_maxiter", 50)}}, max_attempts=1
            )
        except (ModelFittingError, RuntimeError, ValueError) as exc:
            # A typed failure, never a silent fallback to another design.
            raise Error("model_fit_failed", f"GP hyperparameter fit failed: {str(exc)[:1024]}") from exc
    return model, x, data, [str(w.message)[:512] for w in caught]


def propose(request):
    expected = validate(request)  # no heavy imports for unsupported combinations
    from ..api import random_points

    data = [o for o in request.get("observations", []) if o["status"] == "completed"]
    if len(data) < request["config"].get("initial_points", 8):
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
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
    except ImportError as exc:
        raise Error("missing_dependency", "install bo-craft[botorch] and a CPU Torch build") from exc
    p, c = request["problem"], request["config"]
    model, x, selected, warning_messages = model_for(request)
    d, nobj = len(p["parameters"]), len(p["objectives"])
    pending = request.get("pending", [])
    x_pending = torch.tensor([encode(p, o["x"]) for o in pending], dtype=torch.double) if pending else None
    constraints = [(lambda z, index=nobj + i: z[..., index]) for i, _ in enumerate(p.get("constraints", []))]
    sampler = SobolQMCNormalSampler(torch.Size([c.get("mc_samples", 128)]), seed=request["seed"])
    common = {
        "model": model,
        "X_baseline": x,
        "sampler": sampler,
        "X_pending": x_pending,
        "constraints": constraints or None,
    }
    bounds = torch.stack([torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)])
    if expected == "log_nei":
        acq = qLogNoisyExpectedImprovement(
            **common, objective=GenericMCObjective(lambda z, X=None: z[..., 0]), prune_baseline=True
        )
    elif expected == "log_nehvi":
        ref = [
            v * (1 if m["direction"] == "maximize" else -1)
            for m, v in zip(p["objectives"], c["reference_point"])
        ]
        acq = qLogNoisyExpectedHypervolumeImprovement(
            **common,
            ref_point=ref,
            objective=IdentityMCMultiOutputObjective(outcomes=list(range(nobj))),
            prune_baseline=True,
        )
    else:
        from botorch.acquisition.analytic import PosteriorMean
        from botorch.acquisition.cost_aware import InverseCostWeightedUtility
        from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
        from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
        from botorch.acquisition.utils import project_to_target_fidelity
        from botorch.models.cost import AffineFidelityCostModel

        f = p["fidelity"]
        index = [v["name"] for v in p["parameters"]].index(f["parameter"])
        parameter = p["parameters"][index]
        target = (f["target"] - parameter["lower"]) / (parameter["upper"] - parameter["lower"])
        # Cost is defined on normalized fidelity; positivity is validated in specs.
        cost = AffineFidelityCostModel(
            fidelity_weights={index: f["cost_weight"]}, fixed_cost=f["cost_intercept"]
        )
        fixed = FixedFeatureAcquisitionFunction(PosteriorMean(model), d=d, columns=[index], values=[target])
        reduced = torch.stack([torch.zeros(d - 1, dtype=torch.double), torch.ones(d - 1, dtype=torch.double)])
        require(d >= 2, "unsupported_combination", "multifidelity needs at least one design parameter")
        _, current = optimize_acqf(
            fixed,
            reduced,
            1,
            num_restarts=c.get("num_restarts", 4),
            raw_samples=c.get("raw_samples", 128),
            options={"maxiter": c.get("fit_maxiter", 50)},
        )
        acq = qMultiFidelityKnowledgeGradient(
            model,
            num_fantasies=16,
            current_value=current.detach(),
            cost_aware_utility=InverseCostWeightedUtility(cost_model=cost),
            project=lambda z: project_to_target_fidelity(z, d=d, target_fidelities={index: target}),
        )
    inequality = []
    for con in p.get("input_constraints", []):
        coef = []
        idx = []
        offset = 0
        for i, parameter in enumerate(p["parameters"]):
            a = con["coefficients"].get(parameter["name"], 0)
            if a:
                idx.append(i)
                coef.append(-a * (parameter["upper"] - parameter["lower"]))
                offset += a * parameter["lower"]
        inequality.append(
            (
                torch.tensor(idx),
                torch.tensor(coef, dtype=torch.double),
                offset - con["upper"] - con.get("tolerance", 0),
            )
        )
    try:
        points, value = optimize_acqf(
            acq,
            bounds,
            request["q"],
            num_restarts=c.get("num_restarts", 4),
            raw_samples=c.get("raw_samples", 128),
            inequality_constraints=inequality or None,
            options={"maxiter": c.get("fit_maxiter", 50), "batch_limit": 2},
            sequential=expected != "multifidelity_kg",
        )
    except (RuntimeError, ValueError) as exc:
        raise Error("numerical_failure", str(exc)[:1024]) from exc
    return [decode(p, row.tolist()) for row in points.detach()], {
        "phase": "model",
        "acquisition": expected,
        "noise": c.get("noise", "inferred"),
        "training_ids": [o["id"] for o in selected],
        "warnings": warning_messages,
        "acquisition_value": value.detach().reshape(-1).tolist(),
        "precision": "float64",
        "device": "cpu",
        "training_selection": "latest completed suffix",
    }

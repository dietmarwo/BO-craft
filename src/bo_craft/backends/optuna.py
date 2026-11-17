"""Stateless TPE adapter reconstructed from the caller-owned ledger."""

from ..specs import Error, active, require


def distribution(parameter):
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

    kind = parameter["type"]
    if kind == "categorical":
        return CategoricalDistribution(parameter["choices"])
    cls = IntDistribution if kind == "integer" else FloatDistribution
    return cls(parameter["lower"], parameter["upper"], log=parameter.get("log", False))


def propose(request):
    try:
        import optuna
        from optuna.trial import TrialState, create_trial
    except ImportError as exc:
        raise Error("missing_dependency", "install bo-craft[optuna]") from exc
    p, c = request["problem"], request["config"]
    require("fidelity" not in p, "unsupported_combination", "the Optuna adapter does not model fidelities")
    require(
        c.get("acquisition") is None, "unsupported_combination", "Optuna has no BoTorch acquisition option"
    )
    sampler_kind = c.get("sampler", "tpe")
    # Settings that change BoTorch's algorithm must not be silently ignored here.
    botorch_only = ["trust_region", "replication", "outcome_transform", "num_fantasies", "acq_maxiter"]
    require(
        not any(k in c for k in botorch_only),
        "unsupported_combination",
        "selected configuration belongs to the BoTorch backend",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    constraints = p.get("constraints", [])
    if sampler_kind == "gp":
        require(
            not any(k in c for k in ["multivariate", "group", "constant_liar"]),
            "unsupported_combination",
            "multivariate, group and constant_liar are TPE settings",
        )
        require(
            not any(v.get("active_if") for v in p["parameters"]),
            "unsupported_combination",
            "the Optuna GP sampler needs an unconditional space; use TPE",
        )
        require(
            not p.get("input_constraints"),
            "unsupported_combination",
            "the Optuna GP sampler does not handle linear input constraints",
        )
        # Constraint values are read from the trials' system attributes.
        sampler = optuna.samplers.GPSampler(
            seed=request["seed"],
            n_startup_trials=c.get("initial_points", 8),
            deterministic_objective=c.get("noise", "inferred") == "deterministic",
        )
        multivariate = group = None
    else:
        multivariate = c.get("multivariate", True)
        group = c.get("group", multivariate)
        require(not group or multivariate, "unsupported_combination", "group requires multivariate TPE")

        def constraint_values(trial):
            return trial.user_attrs["constraints"]

        sampler = optuna.samplers.TPESampler(
            seed=request["seed"],
            n_startup_trials=c.get("initial_points", 8),
            multivariate=multivariate,
            group=group,
            constant_liar=c.get("constant_liar", True),
            constraints_func=constraint_values if constraints else None,
        )
    study = optuna.create_study(directions=[m["direction"] for m in p["objectives"]], sampler=sampler)
    for o in request.get("observations", []):
        dist = {v["name"]: distribution(v) for v in p["parameters"] if active(v, o["x"])}
        state = (
            TrialState.COMPLETE
            if o["status"] == "completed"
            else (TrialState.PRUNED if o["status"] == "pruned" else TrialState.FAIL)
        )
        values = [o["values"][m["name"]] for m in p["objectives"]] if state == TrialState.COMPLETE else None
        attrs = (
            {"constraints": [o["values"][v["name"]] for v in constraints]} if constraints and values else {}
        )
        system_attrs = {"constraints": attrs["constraints"]} if attrs else {}
        study.add_trial(
            create_trial(
                params=o["x"],
                distributions=dist,
                values=values,
                state=state,
                user_attrs=attrs,
                system_attrs=system_attrs,
            )
        )
    for o in request.get("pending", []):
        dist = {v["name"]: distribution(v) for v in p["parameters"] if active(v, o["x"])}
        study.add_trial(create_trial(params=o["x"], distributions=dist, state=TrialState.RUNNING))
    points = []
    from ..specs import canonical, validate_point

    occupied = {canonical(o["x"]) for o in request.get("observations", []) + request.get("pending", [])}
    for _ in range(256):
        trial = study.ask()
        x = {}
        for parameter in p["parameters"]:
            if not active(parameter, x):
                continue
            name, kind = parameter["name"], parameter["type"]
            if kind == "categorical":
                x[name] = trial.suggest_categorical(name, parameter["choices"])
            elif kind == "integer":
                x[name] = trial.suggest_int(
                    name, parameter["lower"], parameter["upper"], log=parameter.get("log", False)
                )
            else:
                x[name] = trial.suggest_float(
                    name, parameter["lower"], parameter["upper"], log=parameter.get("log", False)
                )
        try:
            validate_point(p, x)
        except Error as exc:
            if exc.code != "input_infeasible":
                raise
            study.tell(trial, state=TrialState.FAIL)
            continue
        if canonical(x) not in occupied:
            occupied.add(canonical(x))
            points.append(x)
        if len(points) == request["q"]:
            diagnostic = {
                "phase": sampler_kind,
                "state": "rebuilt from immutable observations and pending",
                "sampler": "GPSampler" if sampler_kind == "gp" else "TPESampler",
                "seed": request["seed"],
            }
            if sampler_kind == "tpe":
                diagnostic.update(multivariate=multivariate, group=group)
            return points, diagnostic
    raise Error(
        "candidate_exhausted",
        "the Optuna sampler cannot produce distinct feasible candidates in 256 attempts",
    )

"""Single-objective intermediate-result pruning; caller executes cancellation.

A prune decision is policy, not a terminal observation. Retain partial cost and
confirm execution stop before releasing the caller's reservation.
"""

from .specs import require


def should_prune(
    completed,
    intermediate,
    *,
    direction="minimize",
    policy="median",
    startup_trials=5,
    warmup_steps=1,
    seed=0,
):
    import optuna
    from optuna.trial import create_trial

    require(
        direction in ("minimize", "maximize") and policy in ("median", "hyperband"),
        "unsupported_combination",
        "single-objective median/hyperband pruning only",
    )
    require(
        intermediate and all(type(k) is int and k >= 0 for k in intermediate),
        "invalid_observation",
        "nonnegative integer intermediate steps required",
    )
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=warmup_steps)
        if policy == "median"
        else optuna.pruners.HyperbandPruner(min_resource=1, max_resource=64)
    )
    study = optuna.create_study(
        direction=direction, pruner=pruner, sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    for row in completed:
        require(row, "invalid_observation", "empty completed learning curve")
        study.add_trial(create_trial(value=row[max(row)], intermediate_values=row))
    trial = study.ask()
    for step, value in sorted(intermediate.items()):
        trial.report(value, step)
    return {
        "schema_version": "bo.pruning/0.1",
        "should_prune": trial.should_prune(),
        "step": max(intermediate),
        "policy": policy,
        "execution_stopped": False,
    }

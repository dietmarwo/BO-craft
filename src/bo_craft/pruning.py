"""Single-objective intermediate-result pruning; caller executes cancellation.

A prune decision is policy, not a terminal observation. Retain partial cost and
confirm execution stop before releasing the caller's reservation.
"""

import math
import statistics

from .specs import require


def _checked(completed, intermediate, direction):
    require(direction in ("minimize", "maximize"), "unsupported_combination", "single-objective direction")
    require(
        intermediate and all(type(k) is int and k >= 0 for k in intermediate),
        "invalid_observation",
        "nonnegative integer intermediate steps required",
    )
    for row in [intermediate, *completed]:
        require(
            row
            and all(type(k) is int and k >= 0 for k in row)
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in row.values()
            ),
            "invalid_observation",
            "learning curves need nonnegative integer steps and finite values",
        )


def median_decision(completed, intermediate, *, direction="minimize", startup_trials=5, warmup_steps=1):
    """Optuna's MedianPruner rule in pure Python, for clients without Optuna.

    Prune when the running trial's best value so far is worse than the median
    of the completed trials' values at its latest step. Nothing is pruned
    before `startup_trials` completed curves or before `warmup_steps`, or when
    no completed curve reached that step. `tests/test_plan.py` checks it
    against `should_prune(policy="median")`.
    """
    _checked(completed, intermediate, direction)
    step = max(intermediate)
    decision = {
        "schema_version": "bo.pruning/0.1",
        "should_prune": False,
        "step": step,
        "policy": "median",
        "execution_stopped": False,
    }
    if not completed or len(completed) < startup_trials or step < warmup_steps:
        return decision
    at_step = [row[step] for row in completed if step in row]
    if not at_step:
        return decision
    median = statistics.median(at_step)
    best = min(intermediate.values()) if direction == "minimize" else max(intermediate.values())
    decision["median"] = median
    decision["should_prune"] = best > median if direction == "minimize" else best < median
    return decision


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
        require(
            row and all(type(k) is int and k >= 0 for k in row),
            "invalid_observation",
            "completed learning curves need nonnegative integer steps",
        )
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

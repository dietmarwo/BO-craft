#!/usr/bin/env python3
"""Median pruning on learning curves, at equal total evaluation time.

  pruning_sim.py run --arm none|median --seeds 501-530 --output DIR [--workers 2]
  pruning_sim.py cell --arm A --seed S

Hartmann-6 as a learning-curve problem: an evaluation runs 10 steps of one
time unit each, and reports y_k = f(x) + a(x) exp(-k / 2.5) at step k, with
a(x) = 1 + 0.5 sin(sum x). Its observation is the step-10 value. Trials run
one at a time within 400 time units (40 full evaluations); a trial starts
only while 10 units remain. BO-craft's BoTorch backend proposes from the
completed observations; pruned trials enter as `pruned`.

Arms:
- `none` and `median`: BoTorch, without and with pruning. With pruning the
  backend trains on a pruned trial's last reported value (`pruned: last`);
  ignoring pruned trials, the smoke run on seed 99 kept proposing where it
  had just pruned (63 of 76 trials pruned, final regret 1.75 against 0.043).
- `tpe-none` and `tpe-median`: Optuna TPE, which ranks pruned trials by their
  last value; reported, not gated.
`median` applies `bo_craft.pruning.median_decision` after every step
(5 startup trials, 2 warm-up steps); a pruned trial costs only its steps.
Every observation carries its curve as `intermediate`.
Scored on the latent f: the regret of the empirical answer (best completed
observation), and the time at which a completed trial first had a latent
regret of at most 0.1 (censored at 400).
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel
import problems

STEPS, TIME, INITIAL, TARGET = 10, 400.0, 8, 0.1


def curve(x, f):
    a = 1.0 + 0.5 * math.sin(sum(x.values()))
    return {k: f + a * math.exp(-k / 2.5) for k in range(1, STEPS + 1)}


def cell(arm, seed):
    from bo_craft import propose
    from bo_craft.pruning import median_decision

    problem = problems.get("hartmann6")
    bo = panel.bo_problem(problem)
    if arm.startswith("tpe"):
        config = {"backend": "optuna", "initial_points": INITIAL}
    else:
        config = {"backend": "botorch", "noise": "inferred", "initial_points": INITIAL}
        if arm == "median":
            config["pruned"] = "last"
    prune = arm.endswith("median")
    rng = random.Random(seed)
    clock, observations, curves, hit = 0.0, [], [], None
    pruned = 0
    while TIME - clock >= STEPS:
        if len(observations) < INITIAL:
            x = panel.uniform_point(problem, rng)
        else:
            request = {
                "schema_version": "bo.request/0.1",
                "problem": bo,
                "config": config,
                "seed": seed * 1000 + len(observations),
                "q": 1,
                "observations": [
                    {
                        k: o[k]
                        for k in ("id", "trial_id", "x", "status", "values", "seed", "intermediate")
                        if k in o
                    }
                    for o in observations
                ],
            }
            x = propose(request)["candidates"][0]
        f = problem["metrics"](x)["y"]
        full = curve(x, f)
        seen, stopped = {}, None
        for k in range(1, STEPS + 1):
            seen[k] = full[k]
            if (
                prune
                and k < STEPS
                and median_decision(curves, seen, startup_trials=5, warmup_steps=2)["should_prune"]
            ):
                stopped = k
                break
        clock += stopped or STEPS
        i = len(observations)
        base = {
            "id": str(i),
            "trial_id": str(i),
            "x": x,
            "seed": seed * 1000 + i,
            "latent": {"y": f},
            "intermediate": {str(k): v for k, v in seen.items()},
        }
        if stopped:
            pruned += 1
            observations.append({**base, "status": "pruned"})
        else:
            observations.append({**base, "status": "completed", "values": {"y": full[STEPS]}})
            curves.append(dict(seen))
            if hit is None and f - problem["optimum"] <= TARGET:
                hit = clock
    completed = [o for o in observations if o["status"] == "completed"]
    best = min(completed, key=lambda o: o["values"]["y"])["x"]
    return {
        "status": "completed",
        "makespan": hit if hit is not None else TIME,
        "censored": hit is None,
        "pruned": pruned,
        "trials": len(observations),
        "observations": observations,
        "rules": {"empirical": {"regret": panel.score(problem, [best]), "points": [best]}},
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for name in ("--arm", "--seeds", "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--current-src", default="")
    c = sub.add_parser("cell")
    c.add_argument("--arm", required=True)
    c.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "cell":
        import warnings

        warnings.simplefilter("ignore")
        try:
            record = cell(args.arm, args.seed)
        except Exception as exc:  # noqa: BLE001 -- keep failed cells
            record = {"status": "failed", "error": str(exc)[:1000]}
        print(json.dumps(record, allow_nan=False))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def one(seed):
        path = out / f"pruning-{args.arm}-{seed}.json"
        if path.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.current_src:
            env["PYTHONPATH"] = str(Path(args.current_src).resolve())
        command = [sys.executable, __file__, "cell", "--arm", args.arm, "--seed", str(seed)]
        proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=3600)
        value = (
            json.loads(proc.stdout)
            if proc.returncode == 0
            else {"status": "failed", "stderr": proc.stderr[-2000:]}
        )
        value["cell"] = {
            "set": "pruning",
            "problem": "hartmann6",
            "noise": 0,
            "budget": int(TIME),
            "arm": args.arm,
            "seed": seed,
        }
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")

    lo, hi = (int(v) for v in args.seeds.split("-"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, range(lo, hi + 1)))


if __name__ == "__main__":
    main()

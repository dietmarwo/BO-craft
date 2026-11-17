#!/usr/bin/env python3
"""Multifidelity knowledge gradient against target-fidelity-only BO at equal cost.

  mf_panel.py run --problem P --arm A --seeds 501-530 --output DIR [--workers 2]
  mf_panel.py cell --problem P --arm A --seed S

Arms:
- `mf-kg`: BO-craft's multifidelity KG on the problem with its fidelity parameter;
- `target`: BO-craft's BoTorch qLogNEI on the same problem without the fidelity
  parameter, every evaluation at the target fidelity.

Both arms start from the same 4 target-fidelity points (cost 4) and stop
before the declared cost budget would be exceeded. An evaluation at
normalized fidelity s costs cost_intercept + cost_weight * s. Both use
inferred noise (the multifidelity GP fails numerically with deterministic
noise; see the protocol). Scored on the latent target-fidelity function:
- `posterior`: BO-craft's posterior recommendation, projected to the target;
- `empirical`: the best target-fidelity observation.
"""

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel
import problems

BUDGET, INITIAL = 12.0, 4


def cost(problem, x):
    f = problem["fidelity"]
    p = next(v for v in problem["parameters"] if v["name"] == f["parameter"])
    s = (x.get(f["parameter"], f["target"]) - p["lower"]) / (p["upper"] - p["lower"])
    return f["cost_intercept"] + f["cost_weight"] * s


def cell(problem_name, arm, seed):
    from bo_craft import propose, recommend

    problem = problems.get(problem_name)
    fidelity = problem["fidelity"]
    target_only = arm == "target"
    parameters = [
        v for v in problem["parameters"] if not (target_only and v["name"] == fidelity["parameter"])
    ]
    bo = {"schema_version": "bo.problem/0.1", "parameters": parameters, "objectives": problem["objectives"]}
    if not target_only:
        bo["fidelity"] = fidelity
    config = {"backend": "botorch", "noise": "inferred", "initial_points": INITIAL}
    rng = random.Random(seed)
    design = [{"x1": rng.uniform(-5, 10), "x2": rng.uniform(0, 15)} for _ in range(INITIAL)]
    observations, spent, proposals, began = [], 0.0, 0.0, time.monotonic()

    def record(x):
        nonlocal spent
        full = (
            dict(x)
            if target_only or fidelity["parameter"] in x
            else {**x, fidelity["parameter"]: fidelity["target"]}
        )
        point = dict(x) if target_only else full
        latent = problem["metrics"](full)
        observations.append(
            {
                "id": str(len(observations)),
                "trial_id": str(len(observations)),
                "x": point,
                "status": "completed",
                "values": dict(latent),
                "seed": seed * 1000 + len(observations),
                "latent": latent,
                "cost": cost(problem, full),
            }
        )
        spent += cost(problem, full)

    for x in design:
        record(x)
    while True:
        request = {
            "schema_version": "bo.request/0.1",
            "problem": bo,
            "config": config,
            "seed": seed * 1000 + len(observations),
            "q": 1,
            "observations": [
                {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")} for o in observations
            ],
        }
        result = propose(request)
        x = result["candidates"][0]
        if spent + cost(problem, x if not target_only else {}) > BUDGET + 1e-9:
            break
        proposals += result["diagnostics"]["proposal_seconds"]
        record(x)
    request["observations"] = [
        {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")} for o in observations
    ]
    posterior = recommend(request, kind="posterior")["x"]
    at_target = [
        o for o in observations if o["x"].get(fidelity["parameter"], fidelity["target"]) == fidelity["target"]
    ]
    best = min(at_target, key=lambda o: o["values"]["y"])["x"]
    return {
        "status": "completed",
        "seconds": time.monotonic() - began,
        "proposal_seconds": proposals,
        "cost": spent,
        "evaluations": len(observations),
        "target_evaluations": len(at_target),
        "observations": observations,
        "rules": {
            "posterior": {"regret": panel.score(problem, [posterior]), "points": [posterior]},
            "empirical": {"regret": panel.score(problem, [best]), "points": [best]},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for name in ("--problem", "--arm", "--seeds", "--output"):
        r.add_argument(name, required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--current-src", default="")
    c = sub.add_parser("cell")
    c.add_argument("--problem", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "cell":
        import warnings

        warnings.simplefilter("ignore")
        try:
            record = cell(args.problem, args.arm, args.seed)
        except Exception as exc:  # noqa: BLE001 -- keep failed cells
            record = {"status": "failed", "error": str(exc)[:1000]}
        print(json.dumps(record, allow_nan=False))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def one(seed):
        path = out / f"{args.problem}-0-{int(BUDGET)}-{args.arm}-{seed}.json"
        if path.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.current_src:
            env["PYTHONPATH"] = str(Path(args.current_src).resolve())
        command = [
            sys.executable,
            __file__,
            "cell",
            "--problem",
            args.problem,
            "--arm",
            args.arm,
            "--seed",
            str(seed),
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=3600)
            value = (
                json.loads(proc.stdout)
                if proc.returncode == 0
                else {"status": "failed", "stderr": proc.stderr[-2000:]}
            )
        except subprocess.TimeoutExpired:
            value = {"status": "censored", "reason": "watchdog"}
        value["cell"] = {
            "set": "mf",
            "problem": args.problem,
            "noise": 0,
            "budget": int(BUDGET),
            "arm": args.arm,
            "seed": seed,
        }
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")

    lo, hi = (int(v) for v in args.seeds.split("-"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, range(lo, hi + 1)))


if __name__ == "__main__":
    main()

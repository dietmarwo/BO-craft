#!/usr/bin/env python3
"""Virtual-time simulation of an adaptive campaign's proposal look-ahead.

  async_sim.py run --policy legacy|0|1 --seeds 0-9 --output DIR [--workers 2]
  async_sim.py cell --policy P --seed S

It replays experiment-craft's dispatch rule with real BO-craft proposals and a
modelled clock:
- evaluation durations are lognormal (median 10 s, sigma 0.5), drawn from the
  seed and the trial index;
- a proposal job costs 2.3 s of start-up plus 0.5 s and 0.25 s per extra
  candidate (the measurements of 17 November);
- `concurrent` evaluations run at once; one proposal job at a time.

Policy `legacy` proposes `min(remaining, concurrent)` candidates whenever none
is waiting; policy L proposes `min(remaining, free slots + L)`.
"""

import argparse
import concurrent.futures
import heapq
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

CONCURRENT, BUDGET, INITIAL, LEVEL, PROBLEM = 4, 40, 8, 0.1, "hartmann6"


def duration(seed, index):
    return math.exp(math.log(10.0) + 0.5 * random.Random(seed * 7919 + index).gauss(0, 1))


def proposal_cost(q):
    return 2.3 + 0.5 + 0.25 * (q - 1)


def cell(policy, seed):
    from bo_craft import propose, recommend

    problem = problems.get(PROBLEM)
    config = {"backend": "botorch", "initial_points": INITIAL, "noise": "inferred"}
    clock, events, sequence = 0.0, [], 0
    accepted, running, observations, trials = [], {}, [], 0
    proposal_due = None  # (time, candidates)
    while len(observations) < BUDGET:
        # Dispatch accepted candidates into free slots.
        while accepted and len(running) < CONCURRENT:
            running[sequence] = accepted.pop(0)
            heapq.heappush(events, (clock + duration(seed, sequence), "evaluation", sequence))
            sequence += 1
        remaining = BUDGET - trials
        free = CONCURRENT - len(running)
        if not accepted and proposal_due is None and remaining > 0:
            q = min(remaining, CONCURRENT) if policy == "legacy" else min(remaining, free + int(policy))
            if q > 0:
                request = {
                    "schema_version": "bo.request/0.1",
                    "problem": panel.bo_problem(problem),
                    "config": config,
                    "seed": seed * 1000 + trials,
                    "q": q,
                    "observations": [
                        {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")}
                        for o in observations
                    ],
                    "pending": [{"id": f"p{key}", "x": x} for key, x in running.items()],
                }
                candidates = propose(request)["candidates"]
                trials += len(candidates)
                proposal_due = clock + proposal_cost(q)
                heapq.heappush(events, (proposal_due, "proposal", candidates))
        when, kind, payload = heapq.heappop(events)
        clock = when
        if kind == "proposal":
            accepted.extend(payload)
            proposal_due = None
        else:
            x = running.pop(payload)
            latent = problem["metrics"](x)
            i = len(observations)
            observations.append(
                {
                    "id": str(i),
                    "trial_id": str(i),
                    "x": x,
                    "status": "completed",
                    "values": panel.noisy(problem, latent, LEVEL, seed, payload),
                    "seed": seed * 1000 + payload,
                    "latent": latent,
                }
            )
    request = {
        "schema_version": "bo.request/0.1",
        "problem": panel.bo_problem(problem),
        "config": {"backend": "botorch", "noise": "inferred"},
        "seed": seed,
        "q": 1,
        "observations": [
            {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")} for o in observations
        ],
    }
    model = [p["x"] for p in recommend(request, kind="model")["points"]]
    return {
        "status": "completed",
        "makespan": clock,
        "observations": observations,
        "rules": {
            "empirical": {
                "regret": panel.score(problem, panel.empirical(problem, observations)),
                "points": panel.empirical(problem, observations),
            },
            "model": {"regret": panel.score(problem, model), "points": model},
        },
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--current-src", default="")
    c = sub.add_parser("cell")
    c.add_argument("--policy", required=True)
    c.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "cell":
        import warnings

        warnings.simplefilter("ignore")
        try:
            record = cell(args.policy, args.seed)
        except Exception as exc:  # noqa: BLE001 -- keep failed cells
            record = {"status": "failed", "error": str(exc)[:1000]}
        print(json.dumps(record, allow_nan=False))
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def one(seed):
        path = out / f"async-{args.policy}-{seed}.json"
        if path.exists():
            return
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.current_src:
            env["PYTHONPATH"] = str(Path(args.current_src).resolve())
        command = [sys.executable, __file__, "cell", "--policy", args.policy, "--seed", str(seed)]
        proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False, timeout=3600)
        value = (
            json.loads(proc.stdout)
            if proc.returncode == 0
            else {"status": "failed", "stderr": proc.stderr[-2000:]}
        )
        value["cell"] = {
            "set": "async",
            "problem": PROBLEM,
            "noise": LEVEL,
            "budget": BUDGET,
            "arm": f"lookahead-{args.policy}",
            "seed": seed,
        }
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")

    lo, hi = (int(v) for v in args.seeds.split("-"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, range(lo, hi + 1)))


if __name__ == "__main__":
    main()

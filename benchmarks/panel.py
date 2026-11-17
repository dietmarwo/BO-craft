#!/usr/bin/env python3
"""Predeclared benchmark panels with latent scoring.

  panel.py run --protocol P --set NAME --seeds 0-9 --output DIR [--workers 2]
  panel.py cell --protocol P --set NAME --problem PR --noise L --budget B --arm A --seed S

Every cell is a fresh process with a watchdog. Arms of a stratum share the
initial design and the noise stream (common random numbers), so arms are
compared pairwise by seed. The `baseline` source runs a BO-craft checkout
given by --baseline-src (the 0.1.0 code); `current` runs this checkout.
Failed and censored cells are kept.
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
import problems


def uniform_point(problem, rng):
    x = {}
    for p in problem["parameters"]:
        if p["type"] == "continuous":
            x[p["name"]] = rng.uniform(p["lower"], p["upper"])
        elif p["type"] == "integer":
            x[p["name"]] = rng.randint(p["lower"], p["upper"])
        else:
            x[p["name"]] = rng.choice(p["choices"])
    return x


def noisy(problem, latent, level, seed, index):
    """Objective noise with a stream that depends only on seed and evaluation index."""
    values = dict(latent)
    for k, m in enumerate(problem["objectives"]):
        rng = random.Random(seed * 1_000_003 + index * 101 + k)
        values[m["name"]] = latent[m["name"]] + rng.gauss(0.0, level * problem["scale"])
    return values


def feasible(problem, latent):
    return all(latent[c["name"]] <= 0 for c in problem.get("constraints", []))


def score(problem, points):
    """Latent regret of a recommended point (scalar) or set (hypervolume regret)."""
    if not points:
        return None
    latents = [problem["metrics"](x) for x in points]
    if len(problem["objectives"]) == 1:
        latent = latents[0]
        if not feasible(problem, latent):
            return problem["scale"]  # an infeasible answer costs one typical spread
        return latent[problem["objectives"][0]["name"]] - problem["optimum"]
    names = [m["name"] for m in problem["objectives"]]
    front = [tuple(m[n] for n in names) for m in latents if feasible(problem, m)]
    return problem["optimum"] - problems.hypervolume_2d(front, problem["reference_point"])


def empirical(problem, observations):
    """The best observed values (noisy), feasible on observed constraints."""
    usable = [o for o in observations if feasible(problem, o["values"])]
    if not usable:
        return []
    signs = [1 if m["direction"] == "minimize" else -1 for m in problem["objectives"]]
    names = [m["name"] for m in problem["objectives"]]

    def key(o):
        return [s * o["values"][n] for s, n in zip(signs, names)]

    if len(names) == 1:
        return [min(usable, key=lambda o: key(o)[0])["x"]]
    keys = [key(o) for o in usable]
    return [
        o["x"]
        for o, a in zip(usable, keys)
        if not any(all(v <= u for v, u in zip(b, a)) and any(v < u for v, u in zip(b, a)) for b in keys)
    ]


def bo_problem(problem):
    value = {
        "schema_version": "bo.problem/0.1",
        "parameters": problem["parameters"],
        "objectives": problem["objectives"],
    }
    if problem.get("constraints"):
        value["constraints"] = problem["constraints"]
    return value


def arm_config(arm, problem, level):
    config = dict(arm["config"])
    if config["backend"] == "botorch" or config.get("sampler") == "gp":
        config.setdefault("noise", "deterministic" if level == 0 else "inferred")
    if len(problem["objectives"]) > 1 and config["backend"] == "botorch":
        config["reference_point"] = problem["reference_point"]
    return config


def cell(protocol, set_name, problem_name, level, budget, arm_name, seed):
    from bo_craft import propose

    arm = protocol["arms"][arm_name]
    initial = protocol["sets"][set_name].get("initial_points", 8)
    problem = problems.get(problem_name)
    config = arm_config(arm, problem, level)
    if config["backend"] != "random":
        config["initial_points"] = initial
    rng = random.Random(seed)
    design = [uniform_point(problem, rng) for _ in range(initial)]
    observations, proposals, replicates = [], [], 0
    began = time.monotonic()
    for i in range(budget):
        if i < initial:
            x = design[i]
        else:
            request = {
                "schema_version": "bo.request/0.1",
                "problem": bo_problem(problem),
                "config": config,
                "seed": seed * 1000 + i,
                "q": 1,
                "observations": [
                    {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")}
                    for o in observations
                ],
            }
            result = propose(request)
            proposals.append(result["diagnostics"]["proposal_seconds"])
            x = result["candidates"][0]
            replicates += any(result.get("replicate_of", []) or [])
        latent = problem["metrics"](x)
        observations.append(
            {
                "id": str(i),
                "trial_id": str(i),
                "x": x,
                "status": "completed",
                "values": noisy(problem, latent, level, seed, i),
                "seed": seed * 1000 + i,
                "latent": latent,
            }
        )
    record = {
        "status": "completed",
        "seconds": time.monotonic() - began,
        "proposal_seconds": sum(proposals),
        "replicates": replicates,
        "observations": observations,
        "rules": {},
    }
    best_latent = [o["x"] for o in observations if feasible(problem, o["latent"])]
    if len(problem["objectives"]) == 1 and best_latent:
        name = problem["objectives"][0]["name"]
        sign = 1 if problem["objectives"][0]["direction"] == "minimize" else -1
        record["best_sampled_regret"] = score(
            problem, [min(best_latent, key=lambda x: sign * problem["metrics"](x)[name])]
        )
    for rule in arm.get("rules", protocol["sets"][set_name].get("rules", ["empirical"])):
        if rule == "empirical":
            points = empirical(problem, observations)
        else:
            from bo_craft import recommend

            request = {
                "schema_version": "bo.request/0.1",
                "problem": bo_problem(problem),
                "config": {"backend": "botorch", "noise": "deterministic" if level == 0 else "inferred"},
                "seed": seed,
                "q": 1,
                "observations": [
                    {k: o[k] for k in ("id", "trial_id", "x", "status", "values", "seed")}
                    for o in observations
                ],
            }
            try:
                points = [p["x"] for p in recommend(request, kind="model")["points"]]
            except Exception as exc:  # noqa: BLE001 -- a failed rule is a result, not a crash
                record["rules"][rule] = {"error": str(exc)[:500]}
                continue
        record["rules"][rule] = {"regret": score(problem, points), "points": points[:16]}
    return record


def seeds(text):
    lo, hi = (int(v) for v in text.split("-"))
    return list(range(lo, hi + 1))


def run(args):
    protocol = json.loads(Path(args.protocol).read_text())
    spec = protocol["sets"][args.set]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cells = [
        (s["problem"], s["noise"], s["budget"], arm, seed)
        for seed in seeds(args.seeds)
        for s in spec["strata"]
        for arm in spec["arms"]
    ]

    def one(c):
        problem, level, budget, arm, seed = c
        path = out / f"{problem}-{level}-{budget}-{arm}-{seed}.json"
        if path.exists():
            return
        source = protocol["arms"][arm].get("source", "current")
        env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if source == "baseline":
            env["PYTHONPATH"] = str(Path(args.baseline_src).resolve())
        command = [sys.executable, __file__, "cell", "--protocol", args.protocol, "--set", args.set]
        command += ["--problem", problem, "--noise", str(level), "--budget", str(budget)]
        command += ["--arm", arm, "--seed", str(seed)]
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=args.watchdog, env=env, check=False
            )
            value = (
                json.loads(proc.stdout)
                if proc.returncode == 0
                else {"status": "failed", "stderr": proc.stderr[-4000:]}
            )
        except subprocess.TimeoutExpired:
            value = {"status": "censored", "reason": "watchdog", "watchdog": args.watchdog}
        value["cell"] = {
            "set": args.set,
            "problem": problem,
            "noise": level,
            "budget": budget,
            "arm": arm,
            "seed": seed,
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        temp.replace(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, cells))
    print(json.dumps({"set": args.set, "cells": len(cells)}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--protocol", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--watchdog", type=int, default=1800)
    r.add_argument("--baseline-src", default="")
    c = sub.add_parser("cell")
    c.add_argument("--protocol", required=True)
    c.add_argument("--set", required=True)
    c.add_argument("--problem", required=True)
    c.add_argument("--noise", type=float, required=True)
    c.add_argument("--budget", type=int, required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
        return
    import warnings

    warnings.simplefilter("ignore")
    protocol = json.loads(Path(args.protocol).read_text())
    try:
        record = cell(protocol, args.set, args.problem, args.noise, args.budget, args.arm, args.seed)
    except Exception as exc:  # noqa: BLE001 -- keep arbitrary failed cells
        import traceback

        record = {"status": "failed", "error": str(exc)[:1000], "traceback": traceback.format_exc()[-4000:]}
    print(json.dumps(record, allow_nan=False, default=float))


if __name__ == "__main__":
    main()

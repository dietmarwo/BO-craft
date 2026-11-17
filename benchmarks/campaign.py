#!/usr/bin/env python3
"""Bounded synthetic confirmation panel; latent truth avoids selection on noise."""

import argparse
import concurrent.futures
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def cell(method, problem, seed, budget=12):
    from bo_craft import propose, recommend
    from bo_craft.specs import Error

    d = 20 if problem == "ackley20" else 1
    parameters = [{"name": f"x{i}", "type": "continuous", "lower": -2.0, "upper": 2.0} for i in range(d)]
    p = {
        "schema_version": "bo.problem/0.1",
        "parameters": parameters,
        "objectives": [{"name": "score", "direction": "minimize"}],
    }
    if problem == "constrained":
        p["constraints"] = [{"name": "g"}]
    config = {
        "backend": method,
        "initial_points": 4,
        "noise": "known",
        "fit_maxiter": 8,
        "num_restarts": 2,
        "raw_samples": 32,
        "mc_samples": 32,
        "threads": 1,
    }
    rng = random.Random(seed)
    observations = []
    began = time.monotonic()
    proposal_seconds = 0

    def truth(x):
        if problem == "ackley20":
            import math

            v = list(x.values())
            return (
                -20 * math.exp(-0.2 * (sum(t * t for t in v) / d) ** 0.5)
                - math.exp(sum(math.cos(2 * math.pi * t) for t in v) / d)
                + 20
                + math.e
            )
        return (x["x0"] - 0.23) ** 2

    for i in range(budget):
        request = {
            "schema_version": "bo.request/0.1",
            "problem": p,
            "config": config,
            "seed": seed * 100 + i,
            "q": 1,
            "observations": observations,
        }
        if i < 4:
            x = {v["name"]: rng.uniform(-2, 2) for v in parameters}
        else:
            r = propose(request)
            x = r["candidates"][0]
            proposal_seconds += r["diagnostics"]["proposal_seconds"]
        noise = random.Random(seed * 10000 + i).gauss(0, 0.05)
        values = {"score": truth(x) + noise}
        variance = {"score": 0.0025}
        if problem == "constrained":
            values["g"] = x["x0"] - 0.1
            variance["g"] = 1e-10
        observations.append(
            {
                "id": str(i),
                "trial_id": str(i),
                "x": x,
                "status": "completed",
                "values": values,
                "variances": variance,
                "seed": seed * 10000 + i,
            }
        )
    request["observations"] = observations
    try:
        choice = recommend(request)["observations"][0]
        quality = truth(choice["x"])
        feasible = not p.get("constraints") or choice["values"]["g"] <= 0
    except Error:
        quality = None
        feasible = False
    return {
        "method": method,
        "problem": problem,
        "seed": seed,
        "budget": budget,
        "latent_recommendation_loss": quality,
        "feasible": feasible,
        "seconds": time.monotonic() - began,
        "proposal_seconds": proposal_seconds,
        "observations": observations,
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path)
    a.add_argument("--cell", nargs=3)
    a.add_argument("--seeds", type=int, default=30)
    args = a.parse_args()
    if args.cell:
        m, p, s = args.cell
        try:
            result = cell(m, p, int(s))
        except Exception as e:  # noqa: BLE001 -- retain arbitrary failed benchmark cells
            import traceback

            result = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(result, allow_nan=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": list(range(101, 101 + args.seeds)),
        "methods": ["random", "botorch", "optuna"],
        "problems": ["noisy", "constrained"],
        "budget": 12,
        "watchdog": 90,
        "workers": 2,
    }
    path = args.output / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(c):
        m, p, s = c
        path = args.output / f"{m}-{p}-{s}.json"
        if path.exists():
            return
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--cell", m, p, str(s)],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            value = (
                json.loads(proc.stdout)
                if proc.returncode == 0
                else {"status": "failed", "stderr": proc.stderr[-4000:]}
            )
        except subprocess.TimeoutExpired:
            value = {"status": "censored", "reason": "watchdog"}
        value["cell"] = {"method": m, "problem": p, "seed": s}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        temp.replace(path)

    cells = [(m, p, s) for s in manifest["seeds"] for p in manifest["problems"] for m in manifest["methods"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, cells))
    print(json.dumps({"cells": len(cells)}))


if __name__ == "__main__":
    main()

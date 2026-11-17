#!/usr/bin/env python3
"""Paired summaries and gate verdicts of benchmark panels.

  summarize.py --protocol P --gate NAME DIR [DIR ...]   one predeclared gate
  summarize.py --protocol P --table SET DIR [DIR ...]   medians per stratum and arm
  summarize.py --verify DIR [DIR ...]                   recompute every recorded score

A ratio compares arm:rule A against arm:rule B on the same seeds: the geometric
mean of regret_A / regret_B with a 95% bootstrap interval (2000 resamples,
seed 0). Regrets are floored at 1e-9 times the problem's scale.
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel
import problems


def load(directories):
    cells = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            cells.append(json.loads(path.read_text()))
    return cells


def stratum(cell):
    c = cell["cell"]
    return (c["problem"], c["noise"], c["budget"])


def regret(cell, rule):
    if cell.get("status") != "completed":
        return None
    value = cell["rules"].get(rule, {}).get("regret")
    if value is None:
        return None
    floor = 1e-9 * problems.get(cell["cell"]["problem"])["scale"]
    return max(value, floor)


def ratio(cells, a, b, key):
    """Paired geometric ratio of A over B within one stratum; None values count as failures."""
    (arm_a, rule_a), (arm_b, rule_b) = a.split(":"), b.split(":")
    by_seed = {}
    for cell in cells:
        if stratum(cell) != key:
            continue
        by_seed.setdefault(cell["cell"]["seed"], {})[cell["cell"]["arm"]] = cell
    logs, missing = [], 0
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if arm_a not in pair or arm_b not in pair:
            continue
        ra, rb = regret(pair[arm_a], rule_a), regret(pair[arm_b], rule_b)
        if ra is None or rb is None:
            missing += 1
            continue
        logs.append(math.log(ra / rb))
    if not logs:
        return {"pairs": 0, "missing": missing}
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(logs, k=len(logs))) for _ in range(2000))
    return {
        "pairs": len(logs),
        "missing": missing,
        "ratio": math.exp(statistics.fmean(logs)),
        "low": math.exp(means[49]),
        "high": math.exp(means[1949]),
        "a_better": sum(v < 0 for v in logs),
        "b_better": sum(v > 0 for v in logs),
    }


def gate(protocol, name, cells):
    spec = protocol["gates"][name]
    keys = sorted({stratum(c) for c in cells if c["cell"]["set"] == spec["set"]})
    if spec.get("strata"):
        keys = [k for k in keys if [k[0], k[1]] in spec["strata"]]
    rows, passed = [], True
    for key in keys:
        r = ratio([c for c in cells if c["cell"]["set"] == spec["set"]], spec["a"], spec["b"], key)
        if "ratio" not in r:
            ok = False
        elif spec["criterion"] == "upper<1":
            ok = r["high"] < 1 and r["missing"] == 0
        else:
            ok = r["high"] <= 1.05 and r["missing"] == 0
        passed &= ok
        rows.append({"stratum": list(key), **r, "pass": ok})
    return {
        "gate": name,
        "a": spec["a"],
        "b": spec["b"],
        "criterion": spec["criterion"],
        "passed": passed,
        "rows": rows,
    }


def table(protocol, set_name, cells):
    cells = [c for c in cells if c["cell"]["set"] == set_name]
    out = []
    for key in sorted({stratum(c) for c in cells}):
        for arm in protocol["sets"][set_name]["arms"]:
            group = [c for c in cells if stratum(c) == key and c["cell"]["arm"] == arm]
            if not group:
                continue
            row = {
                "stratum": list(key),
                "arm": arm,
                "cells": len(group),
                "failed": sum(c.get("status") != "completed" for c in group),
            }
            done = [c for c in group if c.get("status") == "completed"]
            for rule in ("empirical", "model"):
                values = [regret(c, rule) for c in done]
                values = [v for v in values if v is not None]
                if values:
                    row[f"median_{rule}"] = statistics.median(values)
            if done:
                row["median_proposal_seconds"] = statistics.median(c["proposal_seconds"] for c in done)
                row["replicates"] = sum(c.get("replicates", 0) for c in done)
            out.append(row)
    return out


def verify(cells):
    """Recompute scores from recorded points and the empirical rule from observations."""
    checked = 0
    for cell in cells:
        if cell.get("status") != "completed":
            continue
        problem = problems.get(cell["cell"]["problem"])
        for o in cell["observations"]:
            latent = problem["metrics"](o["x"])
            assert all(abs(latent[k] - v) <= 1e-9 * (1 + abs(v)) for k, v in o["latent"].items()), cell[
                "cell"
            ]
        empirical = panel.empirical(problem, cell["observations"])
        assert cell["rules"]["empirical"]["points"] == empirical[:16], cell["cell"]
        for rule, value in cell["rules"].items():
            if "regret" in value:
                again = panel.score(problem, value["points"])
                assert again is not None and abs(again - value["regret"]) <= 1e-9 * (1 + abs(again)), (
                    cell["cell"],
                    rule,
                )
                checked += 1
    return {"verified_scores": checked, "cells": len(cells)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol")
    parser.add_argument("--gate")
    parser.add_argument("--table")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("directories", nargs="+")
    args = parser.parse_args()
    cells = load(args.directories)
    if args.verify:
        print(json.dumps(verify(cells)))
        return
    protocol = json.loads(Path(args.protocol).read_text())
    if args.gate:
        print(json.dumps(gate(protocol, args.gate, cells), indent=1))
    else:
        print(json.dumps(table(protocol, args.table, cells), indent=1))


if __name__ == "__main__":
    main()

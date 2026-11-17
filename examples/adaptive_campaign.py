"""A complete admitted local noisy campaign, without a systemd requirement."""

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from experiment_craft import adaptive
from experiment_craft.runtime import Runtime
from experiment_craft.storage import Store, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["random", "botorch"], default="botorch")
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=False)
    node = base / "node"
    # Node initialization is explicit setup; the adaptive client never invents capacity.
    import subprocess

    subprocess.run(
        [
            str(args.runner.resolve()),
            "--root",
            str(node),
            "node",
            "init",
            "--capacity",
            "4",
            "--allow-commands",
        ],
        check=True,
    )
    runtime = Runtime(args.runner, node)
    assert runtime.call("node", "capabilities")["command_jobs"]
    manifest = {
        "schema_version": "experiment.adaptive/0.1",
        "problem": {
            "schema_version": "bo.problem/0.1",
            "parameters": [{"name": "x", "type": "continuous", "lower": -2.0, "upper": 2.0}],
            "objectives": [{"name": "score", "direction": "minimize"}],
            "evaluator": {"id": "synthetic.sphere-noise", "version": "1"},
        },
        "config": {
            "backend": args.backend,
            "initial_points": 4,
            "noise": "known",
            "fit_maxiter": 10,
            "num_restarts": 2,
            "raw_samples": 32,
            "mc_samples": 32,
            "threads": 1,
        },
        "seed": 41,
        "concurrent": 2,
        "python": sys.executable,
        "proposal_slots": 1,
        "proposal_wall_seconds": 60,
        "evaluator": {
            "kind": "command",
            "argv": [sys.executable, "evaluator.py"],
            "slots": 1,
            "wall_seconds": 10,
            "inputs": [{"file": str(Path(__file__).with_name("noisy_evaluator.py")), "path": "evaluator.py"}],
        },
        "budget": {"evaluations": 8, "jobs": 30, "wall_seconds": 300, "slot_seconds": 1500},
    }
    write_json(base / "manifest.json", manifest)
    adaptive.initialize(base / "manifest.json", base / "campaign", args.runner, node)
    with closing(Store(base / "campaign")) as store:
        result = adaptive.advance(store)
        assert result["status"] == "completed", result
        state = store.read()
        ids = [o["id"] for o in state["observations"]]
        assert len(ids) == len(set(ids)) == 8
        before = store.events()
        assert adaptive.advance(store) == result
        assert store.events() == before
        write_json(
            base / "result.json",
            {
                "status": result,
                "observations": state["observations"],
                "jobs": state["jobs"],
                "answer": json.loads((base / "campaign" / "answer.json").read_text()),
                "no_duplicate_continuation": True,
            },
        )
    print(result)


if __name__ == "__main__":
    main()

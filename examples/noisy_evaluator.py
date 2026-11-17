"""Command evaluator example: synthetic known-mean objective, fresh seeded noise.

The runtime stages trial.json and retains observation.json. This model is a
fixture; it is not an engineering simulation or evidence of physical accuracy.
"""

import json
import math
import random
import time
from pathlib import Path

trial = json.loads(Path("trial.json").read_text())
rng = random.Random(trial["seed"])
values = [float(x) for x in trial["x"].values() if isinstance(x, (int, float))]
mean = sum((x - 0.2) ** 2 for x in values)
time.sleep(0.02 + abs(math.sin(trial["seed"])) * 0.06)
observation = {
    **trial,
    "status": "completed",
    "values": {"score": mean + rng.gauss(0, 0.05)},
    "variances": {"score": 0.0025},
    "variance_kind": "mean",
    "count": 1,
    "provenance": {"model": "synthetic-sphere-noise/1", "latent_mean": mean},
}
Path("observation.json").write_text(json.dumps(observation, allow_nan=False) + "\n")

"""Recompute all recorded medians from the bundled raw confirmation data."""

import gzip
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
with gzip.open(EVIDENCE / "confirmation.jsonl.gz", "rt") as f:
    records = [json.loads(line) for line in f]
summary = json.loads((EVIDENCE / "summary.json").read_text())
assert len(records) == summary["records"]
assert not any("status" in r for r in records)
for row in summary["rows"]:
    data = [
        r["latent_recommendation_loss"]
        for r in records
        if r["method"] == row["method"] and r["problem"] == row["problem"]
    ]
    assert statistics.median(data) == row["median_latent_loss"]
print(json.dumps({"verified_records": len(records), "verified_rows": len(summary["rows"])}))

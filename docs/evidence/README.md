# BO-craft synthetic confirmation

1D synthetic known-mean noisy and constrained examples; 12 calls, 4 shared initial observations; 30 seeds. No broad optimizer superiority claim.

| Problem | Method | Median latent loss | Feasible / 30 | Median seconds | Paired ratio [95% bootstrap] |
|---|---|---:|---:|---:|---:|
| noisy | random | 0.0226352 | 30 | 0.0298 | 1 [1, 1] |
| noisy | botorch | 0.0113474 | 30 | 4.593 | 0.2963 [0.1158, 0.6999] |
| noisy | optuna | 0.0102521 | 30 | 0.1641 | 0.7277 [0.2965, 1.736] |
| constrained | random | 0.163051 | 30 | 0.03053 | 1 [1, 1] |
| constrained | botorch | 0.0176758 | 30 | 6.897 | 0.1452 [0.08991, 0.2369] |
| constrained | optuna | 0.0995714 | 30 | 0.1972 | 0.7056 [0.4237, 1.208] |

Recommendation quality uses the known noiseless mean, not the most favorable noisy observation. Fresh real-world replication is still required. The 20-dimensional panel is a five-seed pilot. Service startup, imports and runtime jobs are measured separately by the integration examples.

These are the 0.1.0 panels. The benchmarks of the 17 November improvements (eight problems, predeclared gates, development and confirmation seeds) are in [plan-2026-11-17](plan-2026-11-17/README.md). Their recorded cells are rechecked by `benchmarks/summarize.py --verify`.

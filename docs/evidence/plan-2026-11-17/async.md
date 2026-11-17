# Protocol: proposal look-ahead (plan Phase 4)

Committed before any run. `benchmarks/async_sim.py` replays experiment-craft's dispatch rule with real BO-craft proposals and a modelled clock.

**Setup:**
- Problem: noisy Hartmann-6 (noise level 0.1), 40 evaluations, 8 initial points, 4 concurrent evaluations, BoTorch with inferred noise.
- Evaluation durations are lognormal with median 10 s and σ = 0.5, drawn from the seed and the dispatch number.
- A proposal job costs 2.3 s of start-up, plus 0.5 s, plus 0.25 s per extra candidate.

**Policies:**
- `legacy`: today's rule, proposing `min(remaining, concurrent)` whenever no candidate is waiting.
- `0` and `1`: propose `min(remaining, free slots + L)`.

**D3, development (seeds 0–9).** Among L ∈ {0, 1}, a policy qualifies if its median model-rule regret is at most 1.05 × legacy's. The qualifying policy with the lowest median makespan is chosen. If none qualifies, legacy stays the documented choice and nothing is confirmed.

**G5, confirmation (seeds 501–530).** The chosen policy against legacy must give a model-rule regret ratio with an upper bound of at most 1.05. The makespan ratio is reported.

**Consequence.** experiment-craft keeps the legacy rule when `proposal_lookahead` is absent. If G5 passes, its guide recommends the chosen value.

## D3 result (development, seeds 0–9, 30 runs, none failed)

| Policy | Median model-rule regret | Median empirical regret | Median makespan (s) |
|---|---:|---:|---:|
| legacy | 0.372 | 0.372 | 117.7 |
| L = 0 | 0.237 | 0.237 | 147.7 |
| L = 1 | 0.430 | 0.440 | 127.6 |

- Only L = 0 qualifies; L = 1's median regret is 1.16 × legacy's. So L = 0 is chosen as `lookahead-chosen`.
- It is slower: every candidate waits for a proposal computed with the latest observations.
- G5 is confirmed on seeds 501–530.

## G5 result (confirmation, seeds 501–530, 60 runs, none failed)

| Policy | Median model-rule regret | Median empirical regret | Median makespan (s) |
|---|---:|---:|---:|
| legacy | 0.426 | 0.426 | 130.6 |
| L = 0 | 0.214 | 0.214 | 160.2 |

- Paired ratio of model-rule regret, L = 0 against legacy: **0.42 [0.28, 0.64]**. L = 0 was better in 24 of 30 seeds. G5 (upper bound at most 1.05) **passes**, and the interval lies wholly below 1.
- Makespan ratio: 1.23 [1.22, 1.24]. Waiting for proposals with the latest observations makes this modelled campaign 23% longer.
- **Consequence:** experiment-craft's guide recommends `proposal_lookahead: 0` when evaluations are scarcer than wall time. Without the key, the original rule still applies.
- **Scope:** one problem, four slots, a modelled clock (lognormal durations with median 10 s, 2.8 s per proposal job). With shorter evaluations, the start-up cost weighs more.

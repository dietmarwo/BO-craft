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

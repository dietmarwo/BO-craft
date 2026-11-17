# Benchmark protocol of 17 November 2026

This protocol was committed before any development or confirmation run of the
[improvement plan](../../plans/bo-craft-improvement-plan-2026-11-17.md). Its
machine-readable form is [`benchmarks/protocol-2026-11-17.json`](../../../benchmarks/protocol-2026-11-17.json).
Results are appended below each section after the runs; nothing above the
results changes after a run starts.

## Harness

- `benchmarks/problems.py` defines the problems in pure Python, with known optima.
  - The integer Hartmann optimum, −3.253105230874208, was computed by enumerating the integer grid with multi-start local search.
  - Noise scales are the standard deviation of the objective over 4000 uniform points.
- `benchmarks/panel.py run` runs one fresh process per cell, with a watchdog (1800 s); failed and censored cells are kept.
- Arms of a stratum share the initial design (8 uniform points from the seed) and the noise stream (seed and evaluation index), so arms are compared pairwise by seed.
- Noise is Gaussian on objectives only, with standard deviation = level × scale. Constraints are noise-free.
- BoTorch and Optuna-GP arms use `noise: deterministic` in noise-free strata and `inferred` otherwise.
- `q = 1`; one Torch thread; two cells in parallel on two CPUs.
- Scoring uses the latent function:
  - **empirical:** the best noisy observation, feasible on its observed constraints;
  - **model:** `recommend --kind model` with a BoTorch GP on the arm's data.
  - Regret is the latent value minus the optimum. An infeasible answer costs one scale. Several objectives use hypervolume regret against the reference point.
- `benchmarks/summarize.py` computes:
  - the paired geometric ratio of regrets with a 95% bootstrap interval (2000 resamples, seed 0), with regrets floored at 1e-9 × scale;
  - `--verify`, which recomputes every recorded score and the empirical choices from the observations.

## Development (seeds 0–9) and its selection rules

- **D1, acquisition settings (`acq-dev`).**
  - Settings: s0 = (4 restarts, 128 raw samples, 50 iterations; 0.1.0's), s1 = (8, 256, 100), s2 = (10, 512, 200).
  - For each setting and stratum, take the median model-rule regret.
  - A setting qualifies if its median is at most 1.05 × the best setting's median in every stratum.
  - The qualifying setting with the lowest total median proposal time becomes the default.
  - *Amendment, committed while `acq-dev` was running and before its final table:* if no setting qualifies, s0 (0.1.0's settings) stays the default.
### D1 result (development, seeds 0–9, 120 cells, none failed)

Median model-rule regret:

| Stratum | s0 (4, 128, 50) | s1 (8, 256, 100) | s2 (10, 512, 200) |
|---|---:|---:|---:|
| branin, noise 0, 30 calls | 0.00107 | 0.00283 | 0.00186 |
| branin, noise 0.1, 30 calls | 1.49 | 2.03 | 1.6 |
| hartmann6, noise 0, 60 calls | 0.00219 | 0.00106 | 0.0021 |
| hartmann6, noise 0.1, 60 calls | 0.0739 | 0.0269 | 0.0327 |
| Sum of median proposal seconds per run | 71 | 124 | 175 |

- No setting is within 5% of the best median in every stratum. s0 is best on both Branin strata, s1 on both Hartmann-6 strata (on the noisy one by a factor of 2.7 over s0), and s2 nowhere.
- By the amendment, s0, 0.1.0's settings, stays the default. `acq_maxiter` defaults to 50, the value `fit_maxiter` used to impose on the acquisition.
- With ten seeds the medians are noisy. The larger settings cost 1.7× and 2.5× the proposal time, without a gain in every stratum.

- **D2, replication threshold (`replication-dev`).** τ ∈ {0.5, 1.0}: the threshold with the lower geometric mean, over the strata, of median model-rule regret becomes `botorch-rep`, before any confirmation run.
### D2 result (development, seeds 0–9, 60 cells, none failed)

| Stratum | none: median model regret | τ = 0.5 | τ = 1.0 | replicates proposed (τ = 0.5 / 1.0) |
|---|---:|---:|---:|---:|
| Branin, noise 0.3, 30 calls | 3.01 | 2.46 | 3.01 | 91 / 0 |
| Hartmann-6, noise 0.3, 60 calls | 0.120 | 0.194 | 0.120 | 218 / 0 |

- The rule selects τ = 1.0 (geometric mean 0.601 against 0.691), so `botorch-rep` uses τ = 1.0.
- τ = 1.0 never replicated, and it cannot: the posterior variance of the latent function at an observed point is below the noise variance even with that observation alone, and more data only lowers it. At τ ≥ 1 the policy is identical to no replication.
- **G4 is therefore not run.** Its two arms would be identical, giving a ratio of exactly 1, so G4 is recorded as **failed**.
- **Consequences:**
  - Replication stays opt-in and experimental, with τ = 0.5 as its default.
  - The request schema now refuses `replication_threshold` ≥ 1.
  - The development data for τ = 0.5 are mixed: better on Branin, worse on Hartmann-6.

- **Pilots without gates:** `mixed-dev` (sanity), `tr-pilot` (trust region, Ackley-20), `mo-pilot` (Branin–Currin), `transform-pilot` (Rosenbrock-8). Their capabilities stay experimental.

## Confirmation (fresh seeds 501–530) and its gates

| Gate | Comparison | Strata | Pass if |
|---|---|---|---|
| G1 | `botorch:model` vs `botorch:empirical` | the three noisy strata of `phase1-confirm` | the upper bound is below 1 in each |
| G2 | `botorch:empirical` vs `botorch-0.1.0:empirical` | every stratum of `phase1-confirm` | the upper bound is at most 1.05 in each |
| G3a | `botorch:empirical` vs `tpe:empirical` | every stratum of `mixed-confirm` | the upper bound is below 1 in each |
| G3b | `optuna-gp:empirical` vs `tpe:empirical` | every stratum of `mixed-confirm` | the upper bound is below 1 in each |
| G4 | `botorch-rep:model` vs `botorch:model` | every stratum of `replication-confirm` | the upper bound is below 1 in each |

A gate also fails if any of its cells failed or was censored.

**Consequences:**
- **G1:** if it passes, the docs recommend `--kind model` for noisy problems. Either way, campaigns record both answers.
- **G2:** if it fails, the 0.1.0 acquisition defaults return.
- **G3a/b:** if both pass, the faster GP route is the documented default for mixed spaces; if one passes, that one; if neither, TPE stays the documented default.
- **G4:** replication stays opt-in. A pass is reported as a measured benefit in high noise; a fail as a negative result.

Asynchronous look-ahead (plan Phase 4) has its own protocol in [`async.md`](async.md).

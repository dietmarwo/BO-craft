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

### Development of the mixed spaces (sanity, seeds 0–9, 120 cells, none failed)

Median empirical regret, with the median proposal seconds per run in brackets:

| Stratum | TPE | BoTorch mixed | Optuna GP |
|---|---:|---:|---:|
| integer Hartmann-6, noise 0, 50 calls | 0.475 (0.4) | 0.141 (30) | 0.137 (7.7) |
| integer Hartmann-6, noise 0.1, 50 calls | 0.428 (0.3) | 0.156 (32) | 0.145 (7.4) |
| categorical Branin, noise 0, 30 calls | 1.35 (0.2) | 0.597 (23) | 1.10 (4.1) |
| categorical Branin, noise 0.1, 30 calls | 2.59 (0.2) | 3.17 (29) | 1.63 (4.2) |

No rule depends on this set. It checks that both GP routes run on every stratum before the confirmation.

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

## Confirmation results

### G1 and G2 (`phase1-confirm`, seeds 501–530, 300 cells, none failed)

| Stratum | 0.1.0, empirical | this branch, empirical | this branch, model | Model vs empirical (G1) | This branch vs 0.1.0 (G2) |
|---|---:|---:|---:|---|---|
| Branin, noise 0, 30 calls | 0.00143 | 0.00143 | 0.00329 | — | 1 [1, 1] |
| Branin, noise 0.1, 30 calls | 1.141 | 1.141 | 1.100 | 1.34 [0.89, 2.10], 9 better / 14 worse | 1 [1, 1] |
| constrained Branin, noise 0.1, 30 calls | 1.463 | 1.463 | 0.809 | 0.64 [0.40, 1.04], 18 / 5 | 1 [1, 1] |
| Hartmann-6, noise 0, 60 calls | 0.00143 | 0.00143 | 0.00152 | — | 1 [1, 1] |
| Hartmann-6, noise 0.1, 60 calls | 0.0256 | 0.0256 | 0.0222 | 0.97 [0.84, 1.11], 11 / 10 | 1 [1, 1] |

The first three value columns are median latent regrets. The ratios are paired geometric means with 95% bootstrap intervals.

**G1 fails.**
- The model recommendation is not reliably better than the best noisy observation in any noisy stratum.
- It is better in 18 of 30 seeds on constrained Branin, but that interval reaches 1.04. It is worse in the point estimate on plain Branin.
- The plan's 5-seed Hartmann-6 pilot used a noise standard deviation of 0.1, about 0.26 of the scale, against 0.038 here. It suggested a larger gain than this confirmation shows.
- **Consequence:** `--kind model` stays available and documented with this result; it is not recommended as the default answer. Campaigns record both answers, and their validation replicates use the empirical choice unless asked otherwise.

**G2 passes with exact equality.**
- With the defaults kept by D1 and fewer observations than `training_cap`, this branch makes the same proposals as 0.1.0, cell for cell.
- The training rule only differs above the cap, and the acquisition defaults are unchanged.

### G3a and G3b (`mixed-confirm`, seeds 501–530, 360 cells, none failed)

Median empirical regret, with the median proposal seconds per run in brackets, and the paired ratio against TPE:

| Stratum | TPE | BoTorch mixed | vs TPE (G3a) | Optuna GP | vs TPE (G3b) |
|---|---:|---:|---|---:|---|
| integer Hartmann-6, noise 0, 50 calls | 0.355 (0.4) | 0.140 (28) | 0.048 [0.014, 0.15], 25 / 5 | 0.00197 (7.6) | 0.014 [0.004, 0.048], 26 / 4 |
| integer Hartmann-6, noise 0.1, 50 calls | 0.509 (0.3) | 0.030 (33) | 0.079 [0.040, 0.15], 27 / 3 | 0.153 (7.9) | 0.24 [0.13, 0.46], 22 / 8 |
| categorical Branin, noise 0, 30 calls | 1.219 (0.2) | 0.529 (24) | 0.20 [0.088, 0.42], 24 / 4 | 0.674 (4.1) | 0.41 [0.25, 0.65], 23 / 6 |
| categorical Branin, noise 0.1, 30 calls | 2.995 (0.2) | 4.113 (29) | 0.98 [0.63, 1.48], 15 / 14 | 3.438 (4.1) | 1.08 [0.72, 1.64], 13 / 16 |

**Both gates fail**, because of the noisy categorical Branin stratum. There, neither GP route is distinguishable from TPE; their medians are worse and their geometric means are level. In the other three strata, both routes are far ahead of TPE: 2.4 to 73 times lower regret (geometric means), with intervals well below 1.

**Consequence, as declared:** TPE stays the documented default for mixed spaces. Both GP routes are documented as experimental, with this table. The data suggest choosing a GP route when integer or categorical parameters come with little noise. Under noise and a categorical choice, neither GP route is better than TPE here.

## Pilots (development seeds 0–9, no gates)

| Pilot | Arms | Result |
|---|---|---|
| `transform-pilot`: Rosenbrock-8, noise 0, 60 calls | BoTorch with `log` / `bilog` against standardization | regret ratio 0.046 [0.026, 0.083] (log) and 0.052 [0.023, 0.12] (bilog); better in 10/10 seeds |
| `mo-pilot`: Branin–Currin, noise 0.1, 40 calls | BoTorch, Optuna GP, random | model answer against empirical front, hypervolume regret: BoTorch 0.50 [0.35, 0.69] (9/0), Optuna GP 0.80 [0.66, 0.97] |
| `tr-pilot`: Ackley-20, noise 0, 80 calls | BoTorch with and without `trust_region`, TPE, random | trust region against plain BoTorch: 1.04 [0.93, 1.16] (5/5). BoTorch against random: 0.56 [0.50, 0.64]; against TPE: 0.75 [0.67, 0.86] |

**Caveat for `mo-pilot`.** It added noise with Branin's scale (5.2) to both objectives. That is twice the spread of Currin (2.6), so Currin was very noisy. The harness now has per-objective noise scales for later runs.

**Reading.**
- The transforms and the multi-objective model answer are strong pilot results that a fresh-seed confirmation could promote.
- The trust region shows no benefit at this budget and stays experimental.
- Random sampling reaches only about 4 of the 59.4 attainable hypervolume on Branin–Currin with 40 points, so the hypervolume regrets are large for every arm.

## Part 2: multifidelity and pruning (protocol, committed before their runs)

These are the two comparisons that the first part did not build. There is no development set, because nothing is tuned: every setting below is declared here. The confirmation uses fresh seeds 501–530.

Before this protocol was written, one smoke run per arm on seed 99 (outside every development and confirmation range) checked the runners. It changed the design twice; both changes are below:
- Pruned trials now carry their learning curve. With BoTorch ignoring them, the pruning arm kept proposing where it had just pruned: 63 of 76 trials pruned, final regret 1.75 against 0.043 without pruning.
- The multifidelity knowledge gradient failed at 34 observations with NaN gradients from one set of fantasy samples. The backend now retries the acquisition at most twice, rebuilt with fresh samples and restarts, and records the attempts. A request that does not fail is unchanged: a rerun of a recorded `phase1-confirm` cell reproduced it exactly.

### G6: multifidelity knowledge gradient against target-only BO at equal cost

`benchmarks/mf_panel.py`.

**Problems:**
- `branin_mf`: BoTorch's augmented Branin, whose lower fidelities correlate with the target.
- `branin_mf_misleading`: the same at s = 1, but its lower fidelities mirror x1, so they point to the wrong minima.

**Cost and budget:**
- An evaluation costs 0.25 + 0.75 s, so the target fidelity costs 1.
- Both arms start from the same 4 target-fidelity points and stop before 12 cost units would be exceeded.

**Arms:**
- `mf-kg`: BO-craft's multifidelity knowledge gradient with its defaults.
- `target`: qLogNEI on the same problem without the fidelity parameter.
- Both use inferred noise. With `noise: deterministic`, the multifidelity GP failed numerically at 30 and 60 observations (a non-positive-definite covariance) during sizing on seed 0.

**Scoring:** latent target-fidelity regret of the posterior recommendation (gated) and of the best target-fidelity observation (reported).

**Gate:** on `branin_mf`, `mf-kg:posterior` against `target:posterior` must have an upper bound below 1. `branin_mf_misleading` is reported, not gated.

**Consequence:** if G6 passes, the docs report the measured benefit on correlated fidelities and the misleading result next to it. Either way the adapter stays experimental.

### G7: median pruning at equal total evaluation time

`benchmarks/pruning_sim.py`.

**Problem:** Hartmann-6 as learning curves. An evaluation runs 10 steps of one time unit. Step k reports f(x) + a(x)·exp(−k/2.5), with a(x) = 1 + 0.5·sin(Σx); the observation is the step-10 value.

**Arms:** trials run one at a time within 400 units, i.e. 40 full evaluations, with 8 initial random points.
- `none` and `median`: BoTorch with inferred noise.
- `median` applies `median_decision` after each step (5 startup trials, 2 warm-up steps). Pruned trials reach BO-craft as `pruned`, with their curve in `intermediate`; the backend trains on their last value (`pruned: last`).
- `tpe-none` and `tpe-median` repeat this with Optuna TPE, which ranks pruned trials by their last value. They are reported, not gated.

**Scoring:**
- time to the first completed trial with latent regret ≤ 0.1, censored at 400;
- latent regret of the best completed observation.

**Gates:** both must pass:
- G7a: time ratio with an upper bound below 1;
- G7b: regret ratio with an upper bound of at most 1.05.

**Consequence:** if both pass, experiment-craft's guide reports pruning as measured to save time on learning curves; otherwise it reports the result and pruning stays opt-in. Either way, the runner-backed integration test checks that pruning cancels jobs and that the runtime records the cancellation.

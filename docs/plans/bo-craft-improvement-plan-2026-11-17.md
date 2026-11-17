# BO-craft improvement plan — 17 November 2026

**Scope:** [BO-craft](https://github.com/dietmarwo/BO-craft) at `e3dced1` (the reviewed 0.1.0 source, untagged) and its use by [experiment-craft's adaptive campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/adaptive-bo.md) at `62973f5`.

**Basis:**
- a reading of the whole package, about 1,100 lines of Python;
- the review of 16 November (`claude/bo-craft-review-log-2026-11-16.md`);
- fcmaes-rust's comparison of surrogate libraries (`benchmarks/surrogate-comparison/`);
- three new pilot measurements, made for this plan (section 3, appendix).

## 1. Summary

1. **The evidence is thinner than the feature list.** The only confirmation panels are 1-D, with 12 evaluations and a linear constraint. The delivered plan's own benchmark panel (Branin/Hartmann, mixed, multi-objective, multi-fidelity, asynchronous, replication) was never run. **Build the benchmark harness first (Phase 0)** and gate every change on it.
2. **Recommend with the model, not the luckiest observation.** `recommend --kind empirical` returns the best *noisy* observation, and campaigns return no recommendation at all. In the noisy Hartmann-6 pilot, the posterior recommendation beat the empirical one in 5/5 seeds for both GP arms. For BoTorch the median regret fell from 0.21 to 0.14. This is the cheapest real gain (Phase 1).
3. **A GP for mixed spaces.** The BoTorch backend accepts continuous parameters only, so every integer or categorical problem falls back to TPE. In the pilot, TPE was the weakest model-based arm (median regret 0.40 against 0.19–0.21 for the GPs). The pinned BoTorch 0.18.1 already ships `MixedSingleTaskGP` and `optimize_acqf_mixed_alternating`; Optuna 5.0's `GPSampler` handles mixed spaces too (Phase 2).
4. **Don't compete where the native surrogate wins.** On deterministic continuous boxes, fcmaes-rust's surrogate reaches 2–4 orders of magnitude lower regret than BoTorch, at about 1/300 of the proposal time (section 3.3). BO-craft's place is noise, mixed or conditional spaces, unknown constraints, several objectives and fidelities. The docs should route the rest to `fcmaes.surrogate/1` or `/2`.
5. **Proposal jobs are mostly start-up.** A fresh `bo-craft propose` with BoTorch takes 2.8 s, of which about 0.45 s is the actual model work; Torch's import alone is 1.4 s. Every adaptive proposal is a new command job. Batch sizing and look-ahead in experiment-craft matter more than speeding up the model (Phase 4).
6. **Small fixes with large effects:**
   - `fit_maxiter` also caps the acquisition optimizer.
   - The acquisition optimizer defaults to 4 restarts and 128 raw samples (typical use: 10 and 512).
   - The training set is the *latest* 256 observations, so a long campaign can drop its own incumbent.
   - Proposals can never replicate a point, although noisy problems sometimes need exactly that.

## 2. Where BO-craft stands

| Area | State | Evidence today |
|---|---|---|
| JSON contracts, typed errors, atomic artifacts | solid | 20 unit tests, CLI error contract |
| BoTorch scalar (qLogNEI), constraints, pending | implemented | two 1-D panels, 30 seeds, 12 calls, reduced settings (`fit_maxiter` 8, 2 restarts, 32 raw samples) |
| Multi-objective qLogNEHVI | implemented | one smoke test |
| Multi-fidelity KG | experimental | one numerical test; never compared with target-only BO |
| Optuna TPE, mixed and conditional | implemented | conditional test, 1-D panel |
| Posterior recommendation | scalar, unconstrained BoTorch only | prediction tests |
| Replicates | `specs.aggregate` utility | unit tests; nothing proposes a replicate |
| Pruning | library function | not wired to any job or campaign |
| Vizier, BO-MCP | optional adapters | one local Vizier smoke; BO-MCP contract test |
| Higher dimensions | none | 20-D Ackley, 5 seeds, 12 calls |
| Adaptive campaigns | experiment-craft, CI-tested | integration test with the real runner |

There is no test that the BoTorch wrapper matches direct BoTorch with the same settings (the delivered plan's gate B2).

## 3. Measurements for this plan

All on the session machine: 2 CPUs, one Torch thread, BoTorch 0.18.1, Optuna 5.0.0, Torch 2.6.0. These are pilots, not confirmations.

### 3.1 What a proposal job costs

| Step | Seconds |
|---|---|
| `import json` (interpreter start) | 0.02 |
| `import bo_craft.api` | 0.11 |
| `import optuna` | 0.16 |
| `import torch` | 1.43 |
| `bo-craft propose`, BoTorch, d = 6, n = 24, fresh process (median of 3) | **2.76** |
| the same proposal inside that process | 2.18 |
| later proposals in a warm process, n = 16–64 | 0.41–0.55 |

So about 2.3 s of each BoTorch proposal job is fixed start-up and first-call cost. The model work grows little up to n = 256 (0.52 s), because `training_cap` and `fit_maxiter` bound it. experiment-craft runs each proposal as a separate command job, under the optimization-jobs model of one process per job.

### 3.2 Noisy Hartmann-6 pilot

- Setup: d = 6; Gaussian noise with standard deviation 0.1, against a range of about 3.3; 40 evaluations, the first 8 random and shared across arms; seeds 1–5; q = 1.
- Scoring: on the latent (noise-free) function.
- `posterior` is BO-craft's posterior-mean recommendation, fitted on each arm's data.

| Arm | Median regret, empirical recommendation | Median regret, posterior recommendation | Posterior better | Proposal seconds per run (median) |
|---|---:|---:|---:|---:|
| random | 1.61 | 2.14 | 0/5 | 0.01 |
| BO-craft TPE | 0.40 | 0.29 | 2/5 | 0.25 |
| Optuna `GPSampler` (direct, not in BO-craft) | 0.19 | 0.14 | 5/5 | 5.9 |
| BO-craft BoTorch, defaults | 0.21 | **0.14** | 5/5 | 18.3 |

Readings:
- **The winner's curse is visible.** In seed 2, every arm's empirical pick is the same lucky initial point (regret 0.188), although BoTorch had sampled better points.
- **Model-based recommendation pays off where the data concentrate** (both GP arms), and hurts on scattered data (random).
- **Two GPs are ahead of TPE.** Five seeds cannot rank BoTorch against Optuna's GP.

### 3.3 What fcmaes-rust's comparison says (deterministic problems, 10 seeds)

From `fcmaes-rust/benchmarks/surrogate-comparison/comparison.md`; each library uses its documented defaults.

| Problem | Calls | fcmaes surrogate | BoTorch, noiseless | BoTorch, default | Optuna GP | BiteOpt |
|---|---:|---:|---:|---:|---:|---:|
| Hartmann-6, median regret | 96 | 5.0e-7 | 2.9e-3 | 2.0e-2 | 1.2e-3 | — |
| Hartmann-6, proposal s per run | 96 | 0.33 | 90 | 115 | 13.5 | — |
| Styblinski–Tang-10, median regret | 128 | 56.7 | — | 87.8 | — | **16.7** |

On smooth deterministic boxes the native surrogate is far ahead. On separable multimodal problems BiteOpt is. BO-craft should not try to win either class; it should say so and route.

## 4. Weak points in the code

| # | Where | Problem | Consequence |
|---|---|---|---|
| W1 | `api.recommend` | `empirical` takes the best noisy observation; `posterior` exists only for unconstrained scalar BoTorch | winner's curse (3.2); constrained, multi-objective and TPE campaigns have no model-based answer |
| W2 | experiment-craft `adaptive.py` | campaigns never call `recommend` and reserve no validation evaluations | a campaign ends with a list of observations, not an answer with evidence |
| W3 | `backends/botorch.py` `validate` | continuous parameters only | integer and categorical problems get TPE |
| W4 | `botorch.py` lines 90, 249, 284 | `fit_maxiter` is also the acquisition optimizer's `maxiter` | the delivered benchmark's `fit_maxiter: 8` also cut acquisition optimization to 8 iterations |
| W5 | `botorch.py` defaults | 4 restarts, 128 raw samples, 128 MC samples | weaker acquisition optimization than common practice; never measured |
| W6 | `model_for` | training set = latest `training_cap` (256) completed observations | after 256 evaluations the incumbent and its region can leave the model and `X_baseline` |
| W7 | `api.propose` | any existing x is `duplicate_candidate` | the optimizer can never re-evaluate a promising point under noise |
| W8 | `model_for` | output transform fixed to `Standardize` | heavy-tailed objectives fit badly; fcmaes-rust saw this on wide-box Rosenbrock and Levy |
| W9 | diagnostics | no fitted hyperparameters, no fit-quality signal | an agent cannot tell a good model from a degenerate one |
| W10 | multi-fidelity | `num_fantasies=16` fixed; synchronous; no constraints | expensive and never compared with target-only BO |
| W11 | `services.vizier` | a new study per request (the study ID includes the request digest) | every proposal re-imports all observations, O(n²) in total, and leaves one study per proposal on the server |
| W12 | experiment-craft proposal sizing | proposes `q = min(remaining, concurrent)` as soon as no candidate is waiting | candidates can wait for up to `concurrent − 1` completions; the effect of this staleness is not measured |
| W13 | tests | no comparison with direct BoTorch | a wrapper change could alter the algorithm silently |

## 5. Positioning: what BO-craft is for

Put a chooser in the README and in optimization-jobs' `docs/bo-craft.md`, citing measured results only:

| Problem | First choice | Why |
|---|---|---|
| Deterministic, continuous box, d ≤ ~10, expensive | `fcmaes.surrogate/1` (or `/2`) | 3.3: lower regret, ~300× cheaper proposals |
| Many cheap evaluations, multimodal or separable | BiteOpt, CMA-ES (fcmaes) | 3.3 |
| Noisy objective, replicates, known variances | BO-craft BoTorch | 3.2; Phase 3 |
| Integer or categorical parameters | BO-craft (mixed GP after Phase 2; TPE until then) | 3.2; Phase 2 |
| Conditional parameters | BO-craft TPE | the only supported route |
| Unknown constraints, several objectives, fidelities | BO-craft BoTorch | the only supported route; evidence from Phase 5 |

The routing stays a documented decision, made before a campaign. Nothing switches backends inside a campaign.

## 6. Plan

Each phase lists the change, the tests and a gate. Gates follow the practice of fcmaes-rust's surrogate work: predeclare and commit the protocol, choose on development seeds, confirm on fresh held-out seeds, and keep every failed or censored cell.

### Phase 0 — Benchmark harness and protocol (first, size M)

**Change:**
- Replace `benchmarks/campaign.py` with a panel runner. It takes problem, noise level, arm, budget, q and seed, and writes one record per cell with all observations. It keeps the existing watchdog, retained failures and offline verifier.
- Problems: fcmaes-rust's `problems.py` set (Branin, Hartmann-3/6, Styblinski–Tang-10, Rosenbrock-8, oscillator) at three noise levels (0, low, high), plus the mixed, constrained, multi-objective and multi-fidelity problems added in later phases.
- Scoring: latent regret of each recommendation rule, time-to-target, proposal seconds, failures.
- Arms:
  - random and BO-craft's current backends;
  - **direct BoTorch** with identical settings (the wrapper-fidelity reference);
  - Optuna `GPSampler` (direct);
  - for deterministic strata only, the fcmaes surrogate through `optimization-run` or its comparison harness.
- A wrapper-fidelity unit test: the same model, seed and acquisition give BO-craft's candidate and direct BoTorch's within a declared tolerance.

**Gate:**
- the protocol file is committed before any confirmation run;
- the current 0.1.0 behaviour is measured on held-out seeds and recorded as the baseline every later phase is compared with.

### Phase 1 — Answers and model hygiene (size M)

1. **Model-based recommendations (W1).** A new kind `model`, keeping `empirical` and `posterior` unchanged:
   - Scalar problems: the observed point with the best posterior mean; with constraints, among points whose probability of feasibility is at least `p` (default 0.95).
   - Several objectives: the observed points that are nondominated in posterior mean.
   - It works for every backend's data, because it fits a BoTorch GP on the observations; TPE data included.
   - The output keeps the evidence label "predicted; fresh validation required".
2. **Campaign answers (W2, experiment-craft).**
   - At completion, write `recommendation.json` with both the empirical and the model-based choice.
   - Optional `validation_replicates: k`: reserve k evaluations of the budget to re-evaluate the recommended point with fresh run seeds.
   - Report their mean and standard error as separate evidence.
3. **Separate `acq_maxiter` (W4)** from `fit_maxiter`. Choose defaults for `num_restarts`, `raw_samples` and `mc_samples` (W5) on development seeds, by a predeclared rule: the cheapest setting within 5% of the best median regret.
4. **Training selection that keeps the incumbent (W6):** the best `k` feasible observations plus the most recent ones, up to `training_cap`. Record the selected IDs and the rule in diagnostics.
5. **Diagnostics (W9):** lengthscales, output scale, noise variance, and the leave-one-out standardized error. Warn when a lengthscale sits at its bound or the LOO error is extreme. JSON only, no model state to load back.
6. **Output transforms (W8):** `outcome_transform: standardize | log | bilog | power`. `Bilog` and `Power` exist in the pinned BoTorch. The transform is fixed per campaign, and the empirical recommendation stays in raw units.

**Tests:**
- the model recommendation with constraints;
- multi-objective fixtures with an exactly known posterior;
- the training-set rule at n > cap;
- the diagnostics schema.

**Gate:**
- On the noisy strata, the model recommendation's paired regret ratio against the empirical one has a 95% CI upper bound below 1 for the BoTorch arm.
- In every stratum, Phase 1 against the Phase 0 baseline has an upper bound at most 1.05. No regression.

### Phase 2 — GP for mixed spaces (size M–L)

**Change:**
- In the BoTorch backend, encode integers as discrete normalized coordinates and categoricals as indices.
- Fit `MixedSingleTaskGP` and optimize with `optimize_acqf_mixed_alternating`; both are in BoTorch 0.18.1.
- Conditional parameters stay TPE-only, and are rejected before any evaluation as today.
- Optionally, an `optuna` option `sampler: gp` for Optuna's `GPSampler` (it supports integers, categoricals and constraints). This is a second GP route with lighter code and lower per-proposal cost (5.9 s against 18.3 s per run in 3.2). Check its multi-objective support at the pinned version; reject what it lacks.

**Panel:**
- Hartmann-6 with three dimensions discretized;
- a categorical-shift Branin, where a category selects an offset;
- a small enumerable mixed problem, for exact truth;
- noise levels as in Phase 0.

**Gate:** better than TPE at equal evaluations (CI upper bound below 1) on the mixed strata, and no regression on continuous strata. If both GP routes pass, keep the cheaper one as the default.

### Phase 3 — Noise and replication (size M)

**Change:**
- An explicit `replication` policy (W7):
  - `none` stays the default;
  - `incumbent` lets a proposal re-evaluate the model's current best observed point when its posterior standard deviation exceeds a set fraction of the spread of the observed values.
- The proposal names it `replicate_of: <observation id>`. experiment-craft gives it a fresh run seed. The GP trains on the individual replicates; `aggregate` stays for known-variance inputs.
- A panel over noise-to-signal ratios, including one where the noise dominates.

**Gate:** on the high-noise stratum, latent regret of the model recommendation improves (CI upper bound below 1); no regression elsewhere.

**Contract note:** additive request and proposal fields are fine under `bo.request/0.1` and `bo.proposal/0.1` while nothing is tagged. After the first tag, they need a new schema version.

### Phase 4 — Campaign throughput, with experiment-craft (size M)

**Change:**
- An explicit `proposal_lookahead` in the adaptive manifest:
  - `0` proposes exactly the free evaluation slots;
  - today's behaviour corresponds to `concurrent − 1`.
- Measure both on a virtual-time simulation with variable evaluation durations, as in the delivered plan's asynchronous panel. Count the 2.3 s start-up of each proposal job.
- Keep proposal jobs as isolated command jobs. Warm proposal processes would need a long-lived service, which this ecosystem deliberately avoids.

**Gate:**
- choose the default by time-to-target at equal quality on development seeds;
- confirm on held-out seeds that the chosen look-ahead does not raise the regret of the final recommendation (upper bound at most 1.05).

### Phase 5 — Higher dimensions, several objectives, fidelities (size L)

- **Higher dimensions:** an explicit `trust_region: true` option for BoTorch (a TuRBO-1-style region, as the native surrogate already has).
  - Panel: 20–50 dimensions and 100–500 evaluations, against BiteOpt, CMA-ES and the native surrogates where the objective is deterministic.
  - SAAS needs Pyro, which is not a dependency; consider it only if this panel shows a gap that it targets.
- **Several objectives:** noisy Branin–Currin and DTLZ2.
  - Score the hypervolume of the posterior and the validated front, not of the observed one.
  - Compare against direct qLogNEHVI.
- **Fidelities (W10):** make `num_fantasies` configurable, and compare against target-fidelity-only BO at equal total cost, on a correlated pair and on a deliberately misleading pair (the delivered plan's B9 gate).
- **Gate:** each capability keeps its "experimental" label until its own stratum passes. Failed results are recorded as negative results in the capability matrix.

### Phase 6 — Adapters and pruning (size S–M)

- **Vizier (W11):** one study per campaign, with append-only import keyed by observation ID and digest (the identity check already exists). A diverged study stays `study_diverged`. Gate: a local Vizier test over 50 proposals imports each observation once.
- **BO-MCP:** a decision for the owner, either keep the read-only experimental bridge or remove it. It has no live acceptance, and the delivered log saw no idempotent generation API.
- **Pruning:** wiring it needs intermediate metrics from evaluation jobs, which optimization-jobs does not stream. Write a short design note with optimization-jobs; implement nothing until that exists.

## 7. Benchmark protocol

- Seeds:
  - development seeds 0–9;
  - held-out seeds 501–530, fresh for each confirmation;
  - the same initial design per seed for all arms that share an initial-design size.
- Statistic: the paired geometric ratio of latent regret against the reference arm, with a 95% bootstrap interval per stratum (problem × noise × budget). Report medians and failure counts too.
- Promotion:
  - an improvement claim needs an upper bound below 1 in its targeted strata;
  - a default change needs an upper bound of at most 1.05 in every stratum.
- Cost: proposal seconds and the end-to-end campaign time are reported next to every quality number, including the start-up of proposal jobs.
- Records: raw observations per cell, a verifier that recomputes every summary, and the protocol committed before the confirmation run. This follows the existing `verify_evidence.py` practice.

## 8. Order, sizes and releases

| Order | Phase | Size | Repositories |
|---|---|---|---|
| 1 | 0 Harness and protocol | M | BO-craft |
| 2 | 1 Answers and model hygiene | M | BO-craft, experiment-craft (campaign answers) |
| 3 | 2 Mixed-space GP | M–L | BO-craft |
| 4 | 3 Replication | M | BO-craft, experiment-craft |
| 5 | 4 Proposal look-ahead | M | experiment-craft |
| 6 | 5 Dimensions, objectives, fidelities | L | BO-craft |
| 7 | 6 Adapters | S–M | BO-craft |

- A first tagged release, 0.2.0, fits after Phases 0–2. The owner tags; then the schemas freeze at their version.
- experiment-craft moves its `BO_CRAFT_REVISION` pin with each merged phase.
- The chooser in section 5 goes into optimization-jobs at the next coordinated pin move, together with the deferred `ai-context.md` entry.

**Decisions for the owner:**
1. Add Optuna's `GPSampler` as a second GP route (Phase 2), or BoTorch only?
2. Keep or remove the BO-MCP bridge (Phase 6)?
3. Accept the `replicate_of` contract addition (Phase 3)?

## 9. Not planned

- Competing with the native surrogate on deterministic continuous boxes. Route those problems instead (section 5).
- Switching backends automatically inside a campaign, or an automatic method selector.
- A hosted service, warm proposal daemons, GPU support or multi-host fleet work. These are outside BO-craft's charter (`AGENTS.md`).
- Loading pickled or executable model state. Warm starts, if ever needed, use JSON hyperparameters.

## Appendix — pilot data

- Script: `measure.py` (session-local; not committed). Each arm ran in its own process; two ran in parallel on the 2 CPUs.
- The noise draw for evaluation i of seed s is `Random(1000·s + i).gauss(0, 0.1)`.
- The minimum of Hartmann-6 is taken as −3.32237.

Regret per seed on the latent function, with the empirical recommendation (`emp`) and the posterior recommendation (`post`):

| Seed | BoTorch emp | BoTorch post | Optuna GP emp | Optuna GP post | TPE emp | TPE post | Random emp | Random post |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.212 | 0.143 | 0.082 | 0.078 | 0.399 | 0.286 | 2.234 | 2.302 |
| 2 | 0.188 | 0.168 | 0.188 | 0.143 | 0.188 | 0.561 | 0.188 | 0.244 |
| 3 | 0.019 | 0.015 | 0.416 | 0.352 | 0.545 | 0.560 | 1.889 | 2.141 |
| 4 | 0.236 | 0.109 | 0.420 | 0.381 | 0.421 | 0.055 | 1.122 | 1.295 |
| 5 | 0.293 | 0.286 | 0.074 | 0.041 | 0.036 | 0.261 | 1.612 | 2.670 |

Proposal time per BoTorch proposal in these runs: about 0.6 s (n = 8–39, warm process, one thread).

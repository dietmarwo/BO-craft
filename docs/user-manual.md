# User manual

## Choose a method

Use the native fcmaes surrogate for its tested small deterministic continuous-box scope and low overhead. Use BO-craft when statistical noise, mixed/conditional parameters, black-box constraints or multiple objectives justify Python GP/TPE overhead. Compare at equal scientific target and total cost; more supported options do not imply a universally better optimizer.

| Problem | First choice | Evidence |
|---|---|---|
| Deterministic continuous box, d ≤ ~10, expensive | `fcmaes.surrogate/1` or `/2` (fcmaes-rust) | fcmaes-rust's comparison: on Hartmann-6 with 96 calls, median regret 5.0e-7 against 2.0e-2 for default BoTorch, at 0.33 s instead of 115 s of proposals per run |
| Many cheap evaluations, multimodal or separable | BiteOpt or CMA-ES (fcmaes) | the same comparison: on Styblinski–Tang-10, BiteOpt 16.7 against 56.7 (native surrogate) and 87.8 (BoTorch) |
| Noisy objective | BO-craft BoTorch, answered with `recommend --kind model` | BoTorch handles noise in the model. Answer with the empirical choice or `--kind model`: neither was reliably better on fresh seeds (gate G1, [benchmarks of 17 November](evidence/plan-2026-11-17/README.md)). Validate the answer with fresh replicates. |
| Integer or categorical parameters | Optuna TPE by default; a GP route (BoTorch mixed, or Optuna `sampler: gp`) when the noise is small | gates G3a/G3b ([benchmarks of 17 November](evidence/plan-2026-11-17/README.md)): both GP routes are 2.4–73× better than TPE on integer Hartmann-6 and noise-free categorical Branin, but no better on noisy categorical Branin, so the gates fail and TPE stays the default |
| Conditional parameters | BO-craft Optuna TPE | the only supported route |
| Unknown constraints, several objectives, fidelities | BO-craft BoTorch | supported; little evidence beyond the pilots in docs/evidence/plan-2026-11-17 |

Choose before a campaign. Nothing switches the backend inside one.

## Make a request

Define named parameters and objective directions in bo.problem/0.1. Supply evaluator id/version, units and input hashes for real models. Each observation has immutable id/trial_id, named physical x, status, raw values and available uncertainty. Completed observations contain every objective/constraint metric. Failed, cancelled, pruned and unknown work do not become numerical GP targets, with one opt-in exception for pruned trials (below).

For known-noise mode provide variance for every observed metric. An aggregate over independent replications records count and variance_kind=mean. Variance of individual outcomes is not variance of their sample mean. `bo_craft.specs.aggregate` requires distinct replicate seeds and preserves source IDs. A physical model must justify independence; the utility cannot infer it.

Repeated use of the same x and simulation run_seed reproduces a sample path. Use separate run seeds for independent replications. The integrated native evaluate bridge supplies a distinct trial seed and preserves provenance. Its current scope is one unconstrained scalar metric and one thread per evaluation point.

## Spaces

The BoTorch backend handles continuous, integer and categorical parameters:
- continuous parameters on [0, 1], log-scaled when declared;
- integers as their allowed normalized values, at most 4096 values per integer;
- categoricals through a categorical kernel (`MixedSingleTaskGP`), optimized with BoTorch's alternating mixed search.

Conditional parameters (`active_if`) are refused before any evaluation; use Optuna TPE for them. Optuna's GP sampler (`"backend": "optuna", "sampler": "gp"`) is a second GP route for integer and categorical spaces, with constraints and several objectives. It refuses conditional spaces and linear input constraints.

## Settings of the BoTorch backend

| Setting | Default | Meaning |
|---|---|---|
| `initial_points` | 8 | seeded random proposals before the first model |
| `fit_maxiter` | 50 | L-BFGS iterations of the hyperparameter fit |
| `acq_maxiter` | 50 | iterations per acquisition restart (before 17 November, `fit_maxiter` also set this) |
| `num_restarts`, `raw_samples`, `mc_samples` | 4, 128, 128 | acquisition optimization; rule D1 of the benchmark protocol kept 0.1.0's values |
| `training_cap` | 256 | at most this many observations train the model: the best quarter plus the latest |
| `outcome_transform` | `standardize` | or `log` (positive objectives) or `bilog`, applied to the objectives before standardization |
| `replication` | `none` | `incumbent`: see below |
| `trust_region` | false | experimental TuRBO-1 box for scalar, unconstrained, continuous problems |
| `num_fantasies` | 16 | multifidelity knowledge gradient |
| `feasibility_probability` | 0.95 | for `recommend --kind model` with constraints |
| `pruned` | `ignore` | `last`: train on each pruned trial's last intermediate value (one unconstrained objective, no fidelity) |

If the acquisition optimizer fails with NaN gradients, the backend rebuilds the acquisition with fresh Monte Carlo samples and restarts, at most twice; `acquisition_attempts` in the diagnostics counts the attempts. A proposal that does not fail is unchanged.

Diagnostics of every model proposal include the fitted lengthscales, the noise variance and exact leave-one-out residuals in standardized units. A leave-one-out RMS z-score well above 1 means the model is overconfident, and a lengthscale at its lower bound means it found no smooth structure. Both are reported as warnings; neither changes the proposal.

## Learning curves and pruning

A completed or pruned observation of a single-objective problem may carry `intermediate`: its learning curve, a map from step (written as a string) to a finite value. `bo_craft.pruning.median_decision(completed_curves, curve, direction=..., startup_trials=5, warmup_steps=1)` answers whether a running trial should stop: its best value so far is worse than the median of the completed curves at its latest step. It is Optuna's median rule without Optuna.

A pruned trial has no values. The BoTorch backend drops it unless `pruned: last`, which trains on its last intermediate value. Without that, BoTorch keeps proposing where trials were just pruned. TPE receives pruned trials with their curves either way. experiment-craft runs the rule live in adaptive campaigns. On simulated learning curves it showed no time or regret gain (gates G7a/G7b, [note](pruning-and-services.md#pruning)), so use it only where early values rank evaluations reliably.

## Recommendations

- `empirical`: the best observed values, or the observed Pareto set. With noise, this picks the luckiest reading.
- `model`: among the observed points, those with the best posterior mean of the fitted GP. With constraints, only points whose feasibility probability is at least `feasibility_probability` are considered. With several objectives, the result is the points that are nondominated in posterior mean. It fits a BoTorch GP to any backend's data, so it needs the `botorch` extra, and it refuses conditional spaces.
- `posterior`: the minimizer of the posterior mean over the whole space, for scalar unconstrained problems. It is a prediction at a point that may never have been evaluated.

Back-transformed summaries report a median and a 95% interval. A mean is reported only without an outcome transform. In the confirmation of 17 November the model recommendation was not reliably better than the empirical one: its paired regret ratio was 0.97 [0.84, 1.11] on noisy Hartmann-6, 1.34 [0.89, 2.10] on noisy Branin and 0.64 [0.40, 1.04] on noisy constrained Branin (gate G1, [benchmarks of 17 November](evidence/plan-2026-11-17/README.md)). Use it as a second opinion and validate either answer with fresh replicates.

## Replicates

With `replication: incumbent` (BoTorch, scalar, noise not `deterministic`), a proposal re-evaluates the model's current best observed point when two conditions hold:
- its latent standard deviation exceeds `replication_threshold` (default 0.5) times the estimated noise standard deviation;
- it has fewer than `max_replicates` (default 4) evaluations.

The proposal then lists that observation's ID in `replicate_of`. No other repeated point is accepted. The caller must evaluate a replicate with a fresh seed. This policy is experimental. In development, the default threshold 0.5 helped on noisy Branin and hurt on noisy Hartmann-6. A threshold of 1 or more can never trigger, because the latent standard deviation at an observed point is always below the noise level, so the schema refuses it. No confirmation supports replication (gate G4, [benchmarks of 17 November](evidence/plan-2026-11-17/README.md)).

## CLI

```bash
bo-craft validate request.json
bo-craft propose request.json --output proposal.json
bo-craft recommend request.json --kind empirical --output empirical.json
bo-craft recommend request.json --kind model --output model.json
bo-craft recommend request.json --kind posterior --output prediction.json
```

Posterior recommendation currently supports unconstrained scalar BoTorch models, including target-fidelity projection. Its interval is conditional model uncertainty, not an independently validated confidence statement. For constrained and multiobjective problems, use the empirical or the model recommendation.

Check capabilities and typed errors. Conditional BoTorch spaces, missing hypervolume reference points, unsupported fidelity/pending combinations and unknown fields fail before objective evaluation. Numerical failures remain failures. A proposal command's runtime wall limit bounds even backend work that does not support a Python-level interrupt.

## Adaptive campaigns

Install [experiment-craft](https://github.com/dietmarwo/experiment-craft) and this package into one environment; its [adaptive campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/adaptive-bo.md) name the BO-craft revision they are tested with. Start with examples/adaptive_campaign.py, then edit a copy of its manifest. The manifest separates the proposal Python executable/slot/time reservation, evaluator template, scientific evaluation count, concurrency, job count, declared slot-seconds and absolute campaign deadline.

```bash
experiment-craft adaptive-init campaign.json --workspace /absolute/new/campaign \
  --runner /absolute/optimization-run --node-root /absolute/node --mode foreground
experiment-craft adaptive-continue --workspace /absolute/campaign
experiment-craft adaptive-status --workspace /absolute/campaign
experiment-craft adaptive-stop --workspace /absolute/campaign
experiment-craft pause-dispatch --workspace /absolute/campaign --reason maintenance
experiment-craft backup --workspace /absolute/campaign --destination /absolute/backup.zip
```

Use the CLI help for restore/unpause arguments. Foreground mode is a local test path using the existing runtime; it is not a new detached service. Systemd mode requires a commissioned user manager. Generic command jobs currently use the CLI, not optimization-mcp submission tools.

A campaign can have several evaluation jobs and one numerical proposal job in flight. When its optimization budget is spent, it answers with the empirical recommendation, by default (BoTorch) a model recommendation job, and optional fresh validation replicates at the chosen point; see the adaptive guide. The runtime remains CPU capacity authority. Accepted candidates and unknown jobs hold their scientific reservations. Failure/cancellation still consumed work; a declared slot-second reservation is not measured CPU usage and is not refunded speculatively.

A campaign deadline includes time between client sessions. While the client is absent, only native per-job limits are enforced. Unknown submissions are observed by the same handle; never delete/reset the workspace to obtain a retry. A changed source package/executable is refused: finish the old environment or make a deliberate new campaign with warm observations and new provenance.

## Constraints, objectives and fidelities

Input constraints are exact linear physical-space filters for supported active numeric parameters. Unknown observed constraints use g(x)<=0. A simulator crash is not evidence of infeasibility. Model-based feasibility is probabilistic, and every application still needs an independent check.

For multiple objectives supply a raw-unit reference point before the comparison. Keep directions, normalization and reference point fixed. `recommend` reports empirical feasible nondominated observations; noisy posterior fronts need fresh-set validation.

Multifidelity KG is an experimental scalar, continuous, synchronous adapter. Declare normalized-fidelity cost intercept/weight and the raw target fidelity. It currently rejects constraints and pending evaluations. A low-fidelity value is not a target-fidelity answer.

**Prefer target-fidelity BO.** In the confirmation of 17 November (gate G6), at a cost of 0.25 + 0.75 s on augmented Branin, the knowledge gradient spent every evaluation after the initial design below the target fidelity. Its posterior regret was 10.1 [4.8, 21.7] times that of target-only BO at equal cost, and 6 of 30 runs failed numerically. Use it only after comparing it with target-only optimization at equal cost on your own problem. Use inferred noise: with `noise: deterministic` the multifidelity GP failed numerically.

## Optional adapters and reproducibility

Install the pinned Vizier extra only when needed. Configure endpoint/study and a tested algorithm. Each campaign keeps one Vizier study: observations are imported once, Vizier's suggestions are completed by their observations and withdrawn when they fail or are abandoned ([note](pruning-and-services.md#vizier)). The local server tests are not remote outage acceptance. The BO-MCP import bridge expects pre-generated pending suggestions in an existing campaign and does not synchronize results automatically.

Run `pytest`, `ruff check .`, and `python -m build`. The synthetic benchmark harness uses independent watchdogs, two worker processes, immutable cell identities and retained failures. The evidence report includes all 180 scalar confirmation cells and a 15-cell 20D pilot. These are small synthetic examples; they do not establish performance across real engineering experiments or large conditional spaces.
